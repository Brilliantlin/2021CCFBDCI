#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from fuzzywuzzy import fuzz
from tqdm.auto import tqdm
from pandarallel import pandarallel
import collections
from collections import namedtuple
import numpy as np
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
def getRelation(a, b):
    if a in tongyi_table and b in tongyi_table[a]:
        return '同义词'
    if a in fanyi_table and b in fanyi_table[a]:
        return '反义词'
    return '其他'
with open('./data_new/新同义词典.txt','r') as f:
    tongyi = f.readlines()
    tongyi = [x.strip().split(' ')[1:] for x in tongyi]
       
tongyi_table = collections.defaultdict(set)
for l in tongyi:
    l = set(l)
    for w in l:
        tongyi_table[w] = tongyi_table[w] | l
        
#反义词库构建
with open('./data_new/反义词库.txt','r') as f:
    fanyi = f.readlines()
fanyi_table = {}
for l in fanyi:
    a,b = getPair(l)
    fanyi_table.update({a: [b],b:[a]})


test_sub = pd.read_csv('../prediction_result/submission_B_1122.csv',
                       header=None) #xia's model result
me_sub = pd.read_csv(
    '../prediction_result/attention_fgm_tta_0.3_0.3_0.3_vote_2_.csv',
    header=None)  #attnetion fgm 0.3 概率 2票


test = pd.read_csv('../raw_../raw_data/test_B_1118.tsv',sep='\t',header=None)
test.columns = ['text_a','text_b']

# test_a = pd.read_csv('../raw_../raw_data/test_A.tsv',sep='\t',header=None)
# test_a.columns = ['text_a','text_b']


def process(df):
    '''
    将字符串解析成列表
    :param df:
    :return:
    '''
    for col in [
            'deprel_a', 'deprel_b', 'postag_a', 'postag_b', 'word_a', 'word_b'
    ]:
        if isinstance(df[col].iloc[0], str):
            df[col] = df[col].parallel_apply(lambda x: eval(x))
    df['ratio'] = df.parallel_apply(lambda x: fuzz.ratio(x.text_a, x.text_b),
                           axis=1)  #编辑距离
    return df
# data = pd.read_csv('../../data_new/train_eda.csv')
test = pd.read_csv('./data_new/cuted_testB.csv')
# data = process(data)
test = process(test)


def getTokenSet(line):
    c = 'a'
    word = line['word_' + c]
    deps = line['deprel_' + c]
    postag = line['postag_' + c]
    s1 = []
    #     print(word)
    for i in range(len(word)):
        t = Token(i, word[i], deps[i], postag[i])
        s1.append(t)

    c = 'b'
    word = line['word_' + c]
    deps = line['deprel_' + c]
    postag = line['postag_' + c]
    s2 = []
    #     print(word)
    for i in range(len(word)):
        t = Token(i, word[i], deps[i], postag[i])
        s2.append(t)
    return s1, s2


def jugeSwap(line):
    word_a, word_b = line['word_a'], line['word_b']
    postag_a, postag_b = line['postag_a'], line['postag_b']

    if len(word_a) != len(word_b):
        return 'other'
    l = len(word_a)

    # find first diff
    i = 0
    while i < l and word_a[i] == word_b[i]:
        i += 1
    j = i + 1
    media_tmp = ''
    media_tmp_index = []
    #find second diff
    while j < l and word_a[j] == word_b[j]:
        media_tmp += word_a[j]
        media_tmp_index.append(j)
        j += 1
    #只有两个词不一样且第二个词再最后，不用比较后面的部分
    if j == l - 1 and word_a[i] == word_b[j] and word_a[j] == word_b[i]:
        return 'swap:“%s”(%s)交换“%s”(%s),中间词为“%s”' % (
            word_a[i],
            LAC_TABLE.get(postag_a[i], ''),
            word_a[j],
            LAC_TABLE.get(postag_b[j], ''),
            '无' if media_tmp == '' else media_tmp,
        )

    if j < l - 1 and word_a[i] == word_b[j] and word_a[j] == word_b[
            i] and word_a[j + 1:] == word_b[j + 1:]:  #find
        return 'swap:“%s”(%s)交换“%s”(%s),中间词为“%s”' % (
            word_a[i],
            LAC_TABLE.get(postag_a[i], ''),
            word_a[j],
            LAC_TABLE.get(postag_b[j], ''),
            '无' if media_tmp == '' else media_tmp,
        )
    return 'other'


def jugeReplace(line):
    '''
    替换判定
    '''
    s1, s2 = getTokenSet(line)
    v_set1 = set([x[:2] for x in s1])
    v_set2 = set(x[:2] for x in s2)

    s12 = v_set1 - v_set2
    s21 = v_set2 - v_set1
    spb = v_set1 & v_set2

    if len(s12) == 1 and len(s21) == 1 and len(spb) == len(v_set1) - 1 and len(
            spb) == len(v_set2) - 1:  #下标对应，只有一个词语不同
        s1_diff = list(s12)[0][0]
        s2_diff = list(s21)[0][0]

        relation = getRelation(s1[s1_diff].value, s2[s2_diff].value)
        pat = '%s(%s)替换为%s(%s)，他们的关系是%s' % (
            s1[s1_diff].value, LAC_TABLE.get(s1[s1_diff].postag, ''),
            s2[s2_diff].value, LAC_TABLE.get(s2[s2_diff].postag, ''), relation)
        return 'replace:' + pat
    return 'other'


def jugeInsert(line):
    '''
    插入词语判定
    找到第一个不同的单词，删除该单词后判断是否相等
    '''
    word_a, word_b = line['word_a'], line['word_b']
    l1, l2 = len(word_a), len(word_b)
    if abs(l1 - l2) == 1:
        long, short = (word_a, word_b) if l1 > l2 else (word_b, word_a)
        long_postag, short_postag = (line['postag_a'],
                                     line['postag_b']) if l1 > l2 else (
                                         line['postag_b'], line['postag_a'])
        i = 0
        while i < min(l1, l2) and long[i] == short[i]:  #找到第一个不同的单词
            i += 1
        tmp_long = [x for x in long if x != long[i]]
        if tmp_long == short:  #判定为Insert
            #             print('insert ' + long_postag[i],long[i])
            #             print('#',line)
            return 'insert:%s(%s)' % (long[i], LAC_TABLE.get(
                long_postag[i], ''))
    return 'other'



def getNewData(train):
    new_data = []
    for i,line in tqdm(train.iterrows(),total = train.shape[0]):
        line = line.to_dict()
        line['operation'] = 'other'
        
        oper = jugeInsert(line)
        if oper !='other':
            line['operation'] = oper
            
        oper = jugeReplace(line)
        if oper !='other':
            line['operation'] = oper
        
        oper = jugeSwap(line)
        if oper !='other':
            line['operation'] = oper
        new_data.append(line)
    new_df = pd.DataFrame(new_data)
    return new_df


columns = ['index','value','dep','postag']
Token = namedtuple('Token',columns)
testdf = getNewData(test)


## 基于字的交换 、 删除 、 替换

def jugeSwapChar(line):
    '''判断交换'''
    word_a, word_b = line['text_a'], line['text_b']
    if len(word_a) != len(word_b):
        return 'other'
    l = len(word_a)

    # 遍历，找到第一个不相等的位置
    i = 0
    while i < l and word_a[i] == word_b[i]:
        i += 1
    
    # find diff span，所有不相等的字符添加到片段
    diff_word_a1,diff_word_b1 = '',''
    while i < l and word_a[i] != word_b[i]:
        diff_word_a1 += word_a[i]
        diff_word_b1 += word_b[i]
        i += 1
#     print(diff_word_a1,diff_word_b1,i)
    #find median sanme span 中间相同的片段
    media_tmp = ''
    while i < l and word_a[i] == word_b[i]:
        media_tmp += word_a[i]
        i += 1
#     print(media_tmp,i)
    #find second diff 第二段不相等的片段
    diff_word_a2,diff_word_b2 = '',''
    while i < l and word_a[i] != word_b[i]:
        diff_word_a2 += word_a[i]
        diff_word_b2 += word_b[i]
        i += 1
#     print(diff_word_a2,diff_word_b2,i)
    #判断片段1 片段2 是否交换关系，是即返回
    if  diff_word_a1 == diff_word_b2 and diff_word_a2 == diff_word_b1:
        return 'swap:“%s”交换“%s”,中间词为“%s”,后面片段%s' % (
            diff_word_a1,
            diff_word_a2,
            '无' if media_tmp == '' else media_tmp,
            '相等' if word_a[i:] == word_b[i:] else '不相等',
        )
    return 'other'

def jugeInsertChar(line):
    '''
    插入词语判定
    找到第一个不同的单词，删除该单词后判断是否相等
    '''
    word_a, word_b = line['text_a'], line['text_b']
#     word_a = '以太网最短有效帧长'
#     word_b = '以太网有效帧长'
    l1, l2 = len(word_a), len(word_b)
    long, short = (word_a, word_b) if l1 > l2 else (word_b, word_a)

    i = 0
    while i < min(l1, l2) and long[i] == short[i]:  #找到第一个不同的单词
        i += 1
    insert_span = ''
    j = i
    while j < max(l1,l2) and long[j] not in short: #移动长字符串的指针，找出不在short中的片段
        insert_span += long[j]
        j += 1
    if long[j:] == short[i:]:#后面篇片段相等，判定为插入
        return 'insert:%s' % (insert_span)
    return 'other'
def jugeShift(line):
    word_a = line['text_a']
    word_b = line['text_b']
    
    l1, l2 = len(word_a), len(word_b)
    if l1 != l2 : return 'other'
    
    for i in range(1,l1-1):
        if word_a[i:] + word_a[:i] == word_b or word_b[i:] + word_b[:i] == word_a:
            return 'shift'
    return 'other'
    

def getNewDataChar(train):
    new_data = []
    for i,line in tqdm(train.iterrows(),total = train.shape[0]):
        line = line.to_dict()
        line['operation_c'] = 'other'
        oper = jugeInsertChar(line)
        if oper !='other':
            line['operation_c'] = oper

        
        oper = jugeSwapChar(line)
        if oper !='other':
            line['operation_c'] = oper

        oper = jugeShift(line)
        if oper !='other':
            line['operation_c'] = oper
            
        new_data.append(line)
    new_df = pd.DataFrame(new_data)
    return new_df
#直接传入DataFrame 会添加一列operation_c，需要有text_a\text_b两列
testdf_c= getNewDataChar(testdf)
testdf_c['opt'] = testdf_c['operation'].apply(lambda x:x.split(':')[0])
testdf_c['opt_c'] = testdf_c['operation_c'].apply(lambda x:x.split(':')[0])
testdf_c[['text_a','text_b','ratio','operation','opt','operation_c','opt_c']].to_csv('./data_new/test_.csv',index=None)
domain_prob = np.load('../prediction_result/domain_prob.npy')
testdf_c['domain'] = domain_prob.argmax(1)

testdf_c.to_csv('./data_new/new_testB.csv',index = None)



import re
def myReFilter(df,
               text_a=r'.*',
               text_b=r'.*',
               operation=r'.*',
               operation_c=r'.*',
               gate = 0,
               change=False,
               *args,
               **kwargs):
    find = df[(df.text_a.str.contains(text_a) &
               (df.text_b.str.contains(text_b) &
                (df.operation.str.contains(operation) &
                 (df.operation_c.str.contains(operation_c)))))]
    if change:
        for i, line in find[find.label != kwargs['label']].iterrows():
            print(line['text_a'], line['text_b'], line['prob'],line['label'],sep='\t')
        df.loc[(df.text_a.str.contains(text_a) &
                (df.text_b.str.contains(text_b) &
                 (df.operation.str.contains(operation) &
                  (df.operation_c.str.contains(operation_c))))
                ) & ( abs(df.prob - kwargs['label']) > gate), 'label'] = kwargs['label']

    return find


def runRule(df, condiction, debug=False, change=False):
    name = condiction['name']
    label = condiction['label']
    find = myReFilter(df, change=change, **condiction)
    labels = find['label']
    pos, neg = (labels == 1).sum(), (labels == 0).sum()
    info = '%s : 【负样本：%s 正样本:%s change 【%s】 to %s   gate: %s】' % (
        name, neg, pos, neg if label == 1 else pos, label,condiction['gate'] if 'gate' in condiction else 0)
    print(info)
    if debug:
        return find
    return find[find.label != label]

c1 = {
    'name': 'c1',
    'label': 0,
    'operation': r'swap:“.*?”\(地名\)交换“.*?”\(地名\),中间词为“[到|至|去|飞|离]”',
    'text_a': r'.*(.*票$|时刻表$)',
    'gate' : 0.05,
}
c2 = {
    'name':'c2',
    'label':1,
    'operation':
    r'swap:“.*?”\(.*名词\)交换“.*?”\(.*名词\),中间词为“(.*还是.*|.*与.*|.*跟.*|.*既.*|.*同.*|.*及.*|.*?和.*?|.*或.*|×|乘以)”',
    'gate' : 0.05,
}
c3 = {
    'name': 'c3',
    'label': 1,
    'gate' : 0.05,
    'operation': r'swap:“.*?”\(地名\)交换“.*?”\(地名\),中间词为“[到|至|去|飞|离]”',
    'text_a': r'.*(多远|距离|多少公里|多长时间|多久).*'
}
c4 = {
    'name': 'c4',
    'label': 0,
    'gate' : 0.05,
    'operation': r'swap:“.*?”\(.*名词\)交换“.*?”\(.*名词\),中间词为“(.*配.*)”'
}
c5 = {
    'name': 'c5',
    'label': 0,
    'gate' : 0.05,
    'operation': r'swap:“.*?”\(.*\)交换“.*?”\(.*\),中间词为“(.*比)”'
}
c6 = {
    'name': 'c6',
    'label': 0,
    'gate' : 0.05,
    'operation': r'swap:“.*?”\(.*\)交换“.*?”\(.*\),中间词为“(.*转换.*|.*变成.*|.*切换.*)”',
}
c7 = {
    'name': 'c7',
    'label': 0,
    'operation_c': r'swap:“.*?”交换“.*?”,中间词为“[到|至|去|飞|离]”',
    'text_a': r'.*(.*票|时刻表).*',
    'gate' : 0.05,
}
c8 = {
    'name': 'c8',
    'label': 1,
    'gate' : 0.05,
    'operation_c': r'swap:“.*?”交换“.*?”,中间词为“[到|至|去|飞|离]”',
    'text_a': r'.*(多远|距离|多少公里|多长时间|多久).*'
}


## 融合代码

testdf_c['label'] = test_sub[0]
testdf_c['label1'] = test_sub[0]
testdf_c['label2'] = me_sub[0]
testdf_c['xia_prob'] = np.load(
    '../prediction_result/sub_id_B_nezha-large_0.519.npy')
testdf_c['att_prob'] = np.load('../prediction_result/att_prob.npy') #tta 6个 模型的概率平均
testdf_c['mutitask_prob'] = np.load('../prediction_result/mutitask_prob.npy')
testdf_c['prob'] = np.load('../prediction_result/att_prob.npy')
print(testdf_c['label'].sum())


# In[19]:


data = testdf_c
# data['label2'] = pd.read_csv('../../prediction_result/attention_fgm_0.3_0.3_0.3_vote_2_(899).csv',header=None)[0]

for c in [c1,c2,c3,c4,c5,c6,c7,c8]:
    runRule(data,c,debug=True,change = True)
print('swap 处理完毕！')
print(data['label'].sum())

label_col = 'label'
change_index_120 = (data.operation_c.str.contains('insert:(啤|带有|的形状|唐|宋)'))
for i,line in data[change_index_120 & (data[label_col] !=0)][['text_a','text_b',label_col,'operation_c']].iterrows():
    print([x[1] for x in line.to_dict().items()])
data.loc[change_index_120,label_col] = 0
# 11

change_index_021 = (data.operation_c.str.contains('insert:(多远)') & (~data.text_a.str.contains('到')))
for i,line in data[change_index_021 & (data[label_col] !=1)][['text_a','text_b',label_col,'operation_c']].iterrows():
    print([x[1] for x in line.to_dict().items()])
data.loc[change_index_021,label_col] = 1
# 5
print('insert 处理完毕！')
print(data['label'].sum())

cols = {'xia_prob':0.5,'att_prob':0.25,'mutitask_prob':0.25,}
data['rank'] = 0
# rank 排序
for col in cols:
    data[col + '_rank'] = data[col].rank() * cols[col]
    data['rank'] += data[col]
# 概率加权融合
# for col in cols:
#     data['rank'] += data[col] * cols[col]

data['rank'] = data['rank'].rank()
data['rank'] = (data['rank'] - data['rank'].min()) / (data['rank'].max() - data['rank'].min())

ensemble_index =(data.opt == 'other') &                 (data.opt_c == 'other') &                (((data.xia_prob < 0.56) & (data.label1 == 0)) | ((data.xia_prob >= 0.56) & (data.label1 == 1))) &                (data.label1 != data.label2) &                (data.ratio <= 80)
elem = 0.6405
data['label_rank'] = (data['rank'] >= elem).astype(int)
data['tmp'] = data['label'].copy()
print('融合前：',data['tmp'].sum())

data.loc[ensemble_index &
         (data.label_rank != data.tmp), 'tmp'] = data.loc[ensemble_index & (
             data.label_rank != data.tmp), 'label_rank']
#     data.loc[(data.opt_c == 'swap') | (data.opt == 'swap'),'tmp'] = 1 # 藏分操作


# In[20]:


# data['tmp'].to_csv('../../prediction_result/B.csv',index = None,header = None)
# data[(data.opt == 'replace')].sort_values('opt').to_csv('../../../raw_data/34791.csv',index = None)


# In[21]:


def dealnum(s):
    '''将只有一个语文数字的情况统一转换为阿拉伯数字'''
    DIGIT = {
    '一': '1',
    '二': '2',
    '三': '3',
    '四': '4',
    '五': '5',
    '六': '6',
    '七': '7',
    '八': '8',
    '九': '9',
}
    nums = set(['一','二','三', '四','五','六','七','八','九','十'])
    cnt = 0
    for c in s:
        if c in nums:
            cnt += 1
    if cnt == 1:
        news = ''
        for c in s:
            news += DIGIT.get(c,c)
        return news
    return s
def printdf(df):
    for i,line in df.iterrows():
        print(line['text_a'],line['text_b'],line['tmp'],line['operation'],sep='\t')
#数字替换
label_col = 'tmp'
data['deal_text_a'] = data['text_a'].parallel_apply(dealnum)
data['deal_text_b'] = data['text_b'].parallel_apply(dealnum)
change_index = (data['deal_text_a'] == data['deal_text_b']) & (data.tmp != 1)
printdf(data[change_index])
data.loc[change_index,label_col] = 1


# # 同义词纠正

# In[22]:


# with open('../../../raw_data/同义词2.txt','r') as f:
#     tongyi = f.readlines()
#     tongyi = [x.strip().split(' ')[:] for x in tongyi]

# tongyi_table2 = collections.defaultdict(set)
# for l in tongyi:
#     l = set(l)
#     for w in l:
#         tongyi_table2[w] = tongyi_table2[w] | l


# In[23]:


label_col = 'tmp'
cnt = 0
for i ,line in data[data.opt == 'replace'].iterrows():
    operation = line['operation']
    pair = re.findall(r'replace:(.*)\(.*\)替换为(.*)\(.*\).*',operation)
    a,b = pair[0]
    if a in tongyi_table and b in tongyi_table[a] and line['tmp'] !=1 and not re.findall(r'[^形容].*(啥意思|拼音|造句|英文|怎么写的|怎么打字|组什么词|.*?词).*',line['text_a']):
        print(line['text_a'],line['text_b'],line[label_col],line['operation'],sep='\t')
        data.loc[i,label_col] = 1
        cnt += 1


# # 反义词纠正

# In[24]:


label_col = 'tmp'
change_index = data.operation.str.contains('关系是反义词') & (data.tmp == 1)
printdf(data[change_index])
data.loc[change_index,label_col] = 0


# # 组词

# In[26]:


def dealZuci(line):
    a = re.findall(r'的?([^可以怎么写读])[,:字]?怎么(组词|组什么词)$',line['text_a'])
    b = re.findall(r'的?([^可以怎么写读])[,:字]?怎么(组词|组什么词)$',line['text_b'])
    if a and b and a == b:
        print(line['text_a'],line['text_b'],line['tmp'],a,b,sep = '\t')
        return 1
    return line[label_col]
def dealZuci2(line):
    a = re.findall(r'(.*)的([^可以怎么写读]).*?还?能?组什?么?词',line['text_a'])
    b = re.findall(r'(.*)的([^可以怎么写读]).*?还?能?组什?么?词',line['text_b'])
    if a and b:
        a = a[0]
        b = b[0]
#         print(a,b)
        if a[1] and b[1] and (a[1] in a[0]) and (b[1] in b[0]) and (b[1] == a[1]):
            print(line['text_a'],line['text_b'],line['tmp'],a,b,sep = '\t')
            return 1
        if a[1] and b[1] and (a[1] in a[0]) and (b[1] in b[0]) and (b[1] != a[1]):
            print(line['text_a'],line['text_b'],line['tmp'],a,b,sep = '\t')
            return 0
    return line[label_col]
label_col = 'tmp'
data[label_col] = data.apply(dealZuci,axis=1)
data[label_col] = data.apply(dealZuci2,axis=1)


# # 拼音 
# 

# In[27]:


from xpinyin import Pinyin
p = Pinyin()
def pinyin_lin(data, label_col='label', verbose=True):
    '''
    分别获取a\b 两句话的所有发音集合，如有交集，置label为1
    '''
    data['pinyin_c'] = data['text_a'].apply(lambda x: p.get_pinyins(x))
    data['pinyin_d'] = data['text_b'].apply(lambda x: p.get_pinyins(x))
    mohupinyin = data.apply(
        lambda x: len(set(x.pinyin_c) & set(x.pinyin_d)) != 0, axis=1)
    cnt = 0
    for i, line in data[mohupinyin & (data[label_col] != 2) & (
            ~data.text_a.str.contains('(多音字|读音|怎么读|啥意思)'))].iterrows():
        if verbose:
            print(line['text_a'], line['text_b'], line[label_col], sep='\t')
            cnt += 1
    data.loc[mohupinyin &
             (~data.text_a.str.contains('(多音字|读音|啥意思)')) &
             (~data.text_b.str.contains('(多音字|读音|啥意思)')),
             label_col] = 1
    print('change cnt : %s' % (cnt))


pinyin_lin(data, label_col='tmp', verbose=True)
print('save sum', data['tmp'].sum())


data['tmp'].to_csv('../prediction_result/final.csv',index = None,header = None)

