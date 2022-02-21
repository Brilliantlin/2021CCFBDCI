import pandas as pd
import numpy as np
import os
import sys
import warnings
import paddle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
warnings.filterwarnings("ignore")
sys.path.append(SRC_DIR)
#设备配置
DEVICE  = 'gpu:0' if paddle.fluid.is_compiled_with_cuda() else 'cpu'


LAC_TABLE = {
    'n': '普通名词',
    'f': '方位名词',
    's': '处所名词',
    'nw': '作品名',
    'nz': '其他专名',
    'v': '普通动词',
    'vd': '动副词',
    'vn': '名动词',
    'a': '形容词',
    'ad': '副形词',
    'an': '名形词',
    'd': '副词',
    'm': '数量词',
    'q': '量词',
    'r': '代词',
    'p': '介词',
    'c': '连词',
    'u': '助词',
    'xc': '其他虚词',
    'w': '标点符号',
    'PER': '人名',
    'LOC': '地名',
    'ORG': '机构名',
    'TIME': '时间'
}
DEP_TABLE = {
'SBV':	'主谓关系',
'VOB':	'动宾关系'	,
'POB':	'介宾关系',
'ADV':	'状中关系',
'CMP':	'动补关系',
'ATT':	'定中关系',
'F':	'方位关系',
'COO':	'并列关系',
'DBL':	'兼语结构',
'DOB':	'双宾语结构',
'VV':	'连谓结构',
'IC':	'子句结构',
'MT':	'虚词成分',
'HED':	'核心关系',
}

def getLabels2Id(LABELS):
    '''BIO标记法，获得tag:id 映射字典'''
    labels = ['O']
    for label in LABELS:
        labels.append('B-' + label)
        labels.append('I-' + label)
    labels2id = {label: id_ for id_, label in enumerate(labels)}
    id2labels = {id_: label for id_, label in enumerate(labels)}
    return labels2id, id2labels
lac2id,id2lac = getLabels2Id(LAC_TABLE.keys())
dep2id,id2dep = getLabels2Id(DEP_TABLE.keys())