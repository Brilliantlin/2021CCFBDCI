import pickle
import sys
import time
import os
import paddle.nn as nn
import paddle as t
from paddlenlp.transformers import LinearDecayWithWarmup
sys.path.append('../baseline')
def makedirs(prefix):
    '''prefix：文件夹目录，可以递归生成'''
    if not os.path.exists(prefix):
        os.makedirs(prefix)
def savePkl(config, filepath):
    f = open(filepath, 'wb')
    pickle.dump(config, f)
    f.close()

def loadPkl(filepath):
    f = open(filepath, 'rb')
    config = pickle.load(f)
    return config



class BasicModule(nn.Layer):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''
    def __init__(self):
        super(BasicModule,self).__init__()
        self.threshold = 0.5 #默认阈值

    def load(self, path,change_opt=True):
        '''读取模型'''
        data = t.load(path)
        self.set_dict(data)
        return self


    def save(self, name=None,new=False):
        '''存储模型，并返回存储路径'''
        prefix = './user_data/models/'+ self.config.model_name + '/' # 模型前缀
        self.config.save_dir = prefix
        makedirs(prefix) #创建新的目录
        # 模型存储名称
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pdparams')
        path = prefix+name #存储文件名
        data = self.state_dict()
        # print('save path:',path)
        t.save(data, path)
        #save config
        config_dir = './user_data/configs/'
        makedirs(config_dir)
        savePkl(self.config,config_dir +  self.config.model_name + '_'+ 'config.pkl')
        return path

    def getOptimizer(self,learning_rate,max_grad_norm,weight_decay,no_decay = ["bias", "norm"]):
        '''
        :param learning_rate: 学习率，或者 LRshcheduler
        :param max_grad_norm: 全局梯度裁剪
        :param weight_decay: 正则
        :param no_decay: 不需要decay的参数  default ["bias", "norm"]
        :return: optimizer
        '''
        namedparameters = self.named_parameters()

        decay_params = [
            p.name for n, p in namedparameters
            if not any(nd in n for nd in no_decay)
        ]
        if max_grad_norm != None:
            clip = t.nn.ClipGradByGlobalNorm(clip_norm=max_grad_norm)
        else:
            clip = None
        optimizer = t.optimizer.AdamW(
            learning_rate=learning_rate,
            parameters=self.parameters(),
            weight_decay=weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=clip
        )

        return optimizer


