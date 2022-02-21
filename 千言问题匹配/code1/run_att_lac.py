import datetime

import numpy as np

from utils.log_setting import setlog
from utils.dict2Obj import Dict2Obj
from utils.config import loadConfigFromYml
from runconfig import *
configs = loadConfigFromYml('config_add_lac.yaml')
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
from paddlenlp.datasets import MapDataset
from data import create_dataloader, convert_example_with_lac
from model import QuestionMatchingAttentionAddFeature
from train import do_train
import json
from data import QMSet,RawDataNew
import paddlenlp as ppnlp
from runconfig import *
def context():
    paddle.set_device(DEVICE)
    logger.info('set device %s' % DEVICE)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(train_config.seed)
    logger.info(json.dumps(configs.__dict__, indent=1))

TOKEN_MASK_SHAPE = (1,800) #

if __name__ == "__main__":

    #训前的配置
    context()
    rd = RawDataNew()

    train_df = rd.getTrain()
    dev_df = rd.getDev()

    #预训练配置
    tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
        model_config.init_ckpt)
    pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
        model_config.init_ckpt)

    train_ds = MapDataset(QMSet(train_df))
    dev_ds = MapDataset(QMSet(dev_df))

    trans_func = partial(
        convert_example_with_lac,
        tokenizer=tokenizer,
        max_seq_length=data_config.max_seq_length,
        ratio = data_config.ratio)

    def batchify_fn(samples):
        inputs = Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
            Pad(axis=0,pad_val=np.zeros(TOKEN_MASK_SHAPE)), #select_index
            Pad(axis=0,pad_val=0,dtype='int64'),# lac feat
            Pad(axis=0, pad_val=0, dtype='int64'), #dep_feat
            Stack(dtype='int32'), #sequence_length
            Stack(dtype="int64") #label
        )(samples)
        keys = ['input_ids', 'token_type_ids', 'select_tokens','lac_ids','dep_ids','sequence_length','label']
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

    model = QuestionMatchingAttentionAddFeature(pretrained_model, model_config)
    do_train(model,train_config,train_data_loader = train_data_loader,dev_data_loader=dev_data_loader)
