import datetime

import numpy as np

from utils.log_setting import setlog
from utils.dict2Obj import Dict2Obj
from utils.config import loadConfigFromYml
from runconfig import *
from utils.myfile import makedirs

configs = loadConfigFromYml('config_mutitask.yaml')
data_config = Dict2Obj(configs['data_config'])
train_config =  Dict2Obj(configs['train_config'])
model_config = Dict2Obj(configs['model_config'])
configs = Dict2Obj(configs)
logger_name = configs.logger_name
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
setlog.setloger(filename = os.path.join(os.path.join(SRC_DIR, 'logs/[match]%s-%s.log' % (logger_name,now[:-1]))),
                logger_name=logger_name)
logger= setlog.logger
from functools import partial
from utils.seed import set_seed
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset,MapDataset
from data import create_dataloader, convert_example_with_attention_token_domain
from model import QuestionMatchingAttentionDomain
from train import do_train_multitask
import json
from data import RawDataNew,QMSet
import paddlenlp as ppnlp
from runconfig import *
from tqdm.auto import tqdm
import paddle.nn.functional as F
def predict_dev(model, data_loader):
    model.eval()
    batch_logits = []
    total_num = 0
    with paddle.no_grad():
        for batch in tqdm(data_loader, disable=True):
            logits = model(**batch)
            logits = F.sigmoid(logits)
            batch_logits.append(logits.numpy())
    batch_logits = np.concatenate(batch_logits, axis=0)
    preds = batch_logits[:, 1]
    return preds
def context():
    paddle.set_device(DEVICE)
    logger.info('set device %s' % DEVICE)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(train_config.seed)
    logger.info(json.dumps(configs.__dict__, indent=1))

TOKEN_MASK_SHAPE = (1,768)
if __name__ == "__main__":

    #训前的配置
    context()
    data =  pd.read_csv('./data_new/train_eda.csv')
    # dev_df = data.iloc[-28802:,:]
    # train_df = data.iloc[:-28802,:]
    train_df = data
    dev_df = data.head(100)

    #预训练配置
    tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
        model_config.init_ckpt)
    pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
        model_config.init_ckpt)
    train_ds = MapDataset(QMSet(train_df,choice=1))
    dev_ds = MapDataset(QMSet(dev_df,choice=1))

    trans_func = partial(
        convert_example_with_attention_token_domain,
        tokenizer=tokenizer,
        max_seq_length=data_config.max_seq_length,
        ratio = data_config.ratio)

    def batchify_fn(samples):
        inputs = Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
            Pad(axis=0,pad_val=np.zeros(TOKEN_MASK_SHAPE)),
            Stack(dtype = "int64"),
            Stack(dtype="int64"),
            # label
        )(samples)
        keys = ['input_ids', 'token_type_ids', 'select_tokens','domain','label']
        batch_inputs = {k: v for k, v in zip(keys, inputs)}
        return batch_inputs

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=train_config.train_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=train_config.eval_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)
    #模型初始化

    model = QuestionMatchingAttentionDomain(pretrained_model, model_config)
    do_train_multitask(model,train_config,train_data_loader = train_data_loader,dev_data_loader=dev_data_loader,fold_num='single',attack=train_config.attack)

    # logger.info('开始预测保存验证集概率...')
    # model.load('./user_data/models/' + model.config.model_name + '/' + 'best_val_step%s.pdparams' % ('single'))
    # dev_df['pred'] = predict_dev(model, dev_data_loader)
    # makedirs('./user_data/oof/' + model.config.model_name + '/')
    # dev_df.to_csv('./user_data/oof/' + model.config.model_name + '/' + 'fold_num%s_oof.csv' % ('single'))