import gc
import pandas as pd
from pandarallel import pandarallel
from data import RawData
from paddlenlp import Taskflow

pandarallel.initialize(nb_workers = 5)
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
ddp = Taskflow("dependency_parsing", model="ddparser-ernie-gram-zh", use_pos=True)
def getPair(s):
    s = s.strip()
    l = len(s)
    i = 0
    p = ['—', '─', '-', '―']
    a, b = '', ''
    while i < l:
        if s[i] not in p:
            a += s[i]
            i += 1
        else:
            break
    while i < l and s[i] in p:
        i += 1
    b += s[i:]
    return a, b
def convert(df):
    t = pd.DataFrame(ddp(df.text_a.tolist()))
    t.columns = [x + '_a' for x in t.columns]
    print(1)
    t2 = pd.DataFrame(ddp(df.text_b.tolist()))
    t2.columns = [x + '_b' for x in t2.columns]
    df = pd.concat([df.reset_index(drop=True),t.reset_index(drop=True),t2.reset_index(drop=True)],axis=1)
    return df

rd = RawData()
train = rd.getTrain()
dev = rd.getDev()
data = pd.concat([train,dev])

test = pd.read_csv('../raw_data/test_B_1118.tsv',sep='\t',header=None)
test.columns = ['text_a','text_b']




data['text_a'] = data['text_a'].apply(lambda x:x[:100])
data['text_b'] = data['text_b'].apply(lambda x:x[:100])

data = data

data = convert(data)
data.to_csv('./data_new/train_eda.csv',index = None)

test = convert(test)
test.to_csv('./data_new/cuted_testB.csv',index=None)

