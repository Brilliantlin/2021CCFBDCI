import time

import numpy as np
from paddlenlp.transformers import LinearDecayWithWarmup
from tqdm.auto import tqdm
from runconfig import *
import paddle.nn.functional as F
from utils.threshold import getBestThreshold
from utils.log_setting import setlog
from utils.attack import FGM
from sklearn.metrics import  accuracy_score
logger = setlog.logger
@paddle.no_grad()



def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    batch_logits = []
    y_true = []
    total_num = 0
    for batch in data_loader:
        labels = batch['label']
        total_num += len(labels)
        logits = model(**batch)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()

        y_true.append(labels.numpy())
        logits = F.sigmoid(logits)
        batch_logits.append(logits.numpy())

    batch_logits = np.concatenate(batch_logits, axis=0)
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    preds = batch_logits[:, 1]
    thresholdOpt, acc = getBestThreshold(y_true, preds,disable_tqdm=True)

    logger.info("dev_loss: {:.5}, accuracy: {:.5}[{:.5}],threshold:{:.5}, total_num:{}".format(np.mean(losses), accu,acc, thresholdOpt,total_num))
    model.train()
    metric.reset()
    return accu,thresholdOpt


def evaluate_multitask(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    batch_logits = []
    batch_logits_domain = []
    y_true = []
    domain_true = []
    total_num = 0
    for batch in data_loader:
        labels = batch['label']
        domain = batch['domain']
        total_num += len(labels)
        logits,domain_logits = model(**batch)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()

        y_true.append(labels.numpy())
        domain_true.append(domain)
        logits = F.sigmoid(logits)
        domain_logits = domain_logits.argmax(-1)
        batch_logits.append(logits.numpy())
        batch_logits_domain.append(domain_logits.numpy())

    batch_logits = np.concatenate(batch_logits, axis=0)
    batch_logits_domain = np.concatenate(batch_logits_domain,axis = 0)

    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    domain_true = np.concatenate(domain_true,axis = 0).reshape(-1)


    domain_acc = accuracy_score(domain_true,batch_logits_domain)

    preds = batch_logits[:, 1]
    thresholdOpt, acc = getBestThreshold(y_true, preds,disable_tqdm=True)

    logger.info("dev_loss: {:.5}, accuracy: {:.5}[{:.5}],threshold:{:.5}, domain_acc:{:.5},total_num:{}".format(np.mean(losses), accu,acc, thresholdOpt,domain_acc,total_num))
    model.train()
    metric.reset()
    return accu,thresholdOpt

def evaluate_with_threshold(model, data_loader):
    model.eval()
    batch_logits = []
    y_true = []
    total_num = 0
    for batch in tqdm(data_loader):
        labels = batch['label']
        y_true.append(labels.numpy())
        logits = model(**batch)
        logits = F.sigmoid(logits)
        batch_logits.append(logits.numpy())
    batch_logits = np.concatenate(batch_logits, axis=0)
    y_true = np.concatenate(y_true,axis=0).reshape(-1)
    preds = batch_logits[:,1]
    thresholdOpt,acc = getBestThreshold(y_true,preds)
    return thresholdOpt,acc

def do_train(model, train_config, train_data_loader, dev_data_loader, fold_num=0,attack = None ):
    '''

    :param model: 需要训练的模型
    :param train_config: 训练配置
    :return:
    '''
    num_training_steps = len(train_data_loader) * train_config.epochs
    logger.info('总步数 %s' % (num_training_steps))
    lr_scheduler = LinearDecayWithWarmup(train_config.learning_rate, num_training_steps,
                                         train_config.warmup_proportion)
    optimizer = model.getOptimizer(learning_rate=lr_scheduler, max_grad_norm=train_config.max_grad_norm,
                                   weight_decay=train_config.weight_decay)
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    best_accuracy = 0.0
    last_acc = 0.0
    up = False

    tic_train = time.time()
    bar = tqdm(range(1, train_config.epochs + 1), total=train_config.epochs)


    if attack == 'fgm':
        fgm = FGM(model)

    for epoch in bar:
        for step, batch in enumerate(train_data_loader, start=1):
            labels = batch['label']
            logits1 = model(**batch)
            ce_loss = criterion(logits1, labels)
            loss = ce_loss
            loss.backward()

            if attack == 'fgm':
                fgm.attack()
                logits_adv = model(**batch)
                ce_loss = criterion(logits_adv, labels)
                loss = ce_loss
                loss.backward()
                fgm.restore()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()


            # eval
            correct = metric.compute(logits1, labels)
            metric.update(correct)
            acc = metric.accumulate()
            global_step += 1

            if not train_config.silent and global_step % 10 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.4f, ce_loss: %.4f., accu: %.4f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, ce_loss, acc,
                       10 / (time.time() - tic_train)))
                if global_step % train_config.eval_step ==0:
                    logger.info(
                        "【global step %d, epoch: %d, batch: %d】，loss: %.4f, ce_loss: %.4f., accu: %.4f,"
                        % (global_step, epoch, step, loss, ce_loss, acc,))
                tic_train = time.time()


            if global_step % train_config.eval_step == 0 and dev_data_loader !=None:
                accuracy,thresholdOpt = evaluate(model, criterion, metric, dev_data_loader)
                if accuracy > best_accuracy:
                    model.save(name='best_val_step%s.pdparams' % (fold_num))
                    best_accuracy = accuracy
                    model.config.threshold = thresholdOpt
                # if accuracy >= 0.8855:
                #     model.save(name='tmp884.pdparams')
                # if accuracy >= last_acc: #涨分
                #     up = True
                # else:
                #     if up and last_acc >= 0.8855 :# 降分、且当前的分数满足条件
                #         os.rename(model.config.save_dir + 'tmp884.pdparams', model.config.save_dir + 'best_val_step%s_%s.pdparams' % (fold_num,global_step))
                #         logger.info('save ckpt at this step!')
                #     up = False
                # last_acc = accuracy
                if global_step in train_config.save_chkpoint:
                    model.save(name='best_val_step%s_%s.pdparams' % (fold_num,global_step))
            else:
                if global_step >  22000 and global_step % 2000 == 0:
                    model.save(name='best_val_step%s_%s.pdparams' % (fold_num,global_step))

            if global_step == train_config.max_steps:
                logger.info('BEST SCORE ACC: %s ' % best_accuracy)
                return
    model.save(name='best_val_step%s_%s.pdparams' % (fold_num, global_step))
    logger.info('BEST SCORE ACC: %s ' % best_accuracy)
    return




def do_train_multitask(model, train_config, train_data_loader, dev_data_loader, fold_num=0,attack = None ):
    '''

    :param model: 需要训练的模型
    :param train_config: 训练配置
    :return:
    '''
    num_training_steps = len(train_data_loader) * train_config.epochs
    logger.info('总步数 %s' % (num_training_steps))
    lr_scheduler = LinearDecayWithWarmup(train_config.learning_rate, num_training_steps,
                                         train_config.warmup_proportion)
    optimizer = model.getOptimizer(learning_rate=lr_scheduler, max_grad_norm=train_config.max_grad_norm,
                                   weight_decay=train_config.weight_decay)
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    best_accuracy = 0.0
    last_acc = 0.0
    up = False

    tic_train = time.time()
    bar = tqdm(range(1, train_config.epochs + 1), total=train_config.epochs)


    if attack == 'fgm':
        fgm = FGM(model)

    for epoch in bar:
        for step, batch in enumerate(train_data_loader, start=1):
            labels = batch['label']
            domain = batch['domain']
            logits1,domain_logits = model(**batch)
            ce_loss = criterion(logits1, labels)
            domain_loss = criterion(domain_logits, domain)
            loss = ce_loss*0.8 + domain_loss * 0.2
            loss.backward()

            if attack == 'fgm':
                fgm.attack()
                logits_adv,domain_adv = model(**batch)
                ce_loss = criterion(logits_adv, labels)
                domain_loss = criterion(domain_adv, domain)
                loss = ce_loss*0.8 + domain_loss * 0.2
                loss.backward()
                fgm.restore()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()


            # eval
            correct = metric.compute(logits1, labels)
            metric.update(correct)
            acc = metric.accumulate()
            global_step += 1

            if not train_config.silent and global_step % 10 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.4f,domain_loss: %.4f ,ce_loss: %.4f., accu: %.4f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, domain_loss,ce_loss, acc,
                       10 / (time.time() - tic_train)))
                if global_step % train_config.eval_step ==0:
                    logger.info(
                        "【global step %d, epoch: %d, batch: %d】，loss: %.4f,domain_loss: %.4f ,ce_loss: %.4f., accu: %.4f,"
                        % (global_step, epoch, step, loss,domain_loss,ce_loss, acc,))
                tic_train = time.time()


            if global_step % train_config.eval_step == 0 and dev_data_loader !=None:
                accuracy,thresholdOpt = evaluate_multitask(model, criterion, metric, dev_data_loader)
                if accuracy > best_accuracy:
                    model.save(name='best_val_step%s.pdparams' % (fold_num))
                    best_accuracy = accuracy
                    model.config.threshold = thresholdOpt
                if accuracy >= 0.8855:
                    model.save(name='tmp884.pdparams')
                if accuracy >= last_acc: #涨分
                    up = True
                else:
                    if up and last_acc >= 0.8855 :# 降分、且当前的分数满足条件
                        os.rename(model.config.save_dir + 'tmp884.pdparams', model.config.save_dir + 'best_val_step%s_%s.pdparams' % (fold_num,global_step))
                        logger.info('save ckpt at this step!')
                    up = False
                last_acc = accuracy
            else:
                if global_step >  22000 and global_step % 2000 == 0:
                    model.save(name='best_val_step%s_%s.pdparams' % (fold_num,global_step))

            if global_step == train_config.max_steps:
                logger.info('BEST SCORE ACC: %s ' % best_accuracy)
                return
    model.save(name='best_val_step%s_%s.pdparams' % (fold_num, global_step))
    logger.info('BEST SCORE ACC: %s ' % best_accuracy)
    return