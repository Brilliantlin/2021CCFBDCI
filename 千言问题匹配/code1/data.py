
import os
import re
import warnings

import jieba
import numpy as np
import paddle
import pandas as pd
from LAC import LAC
from fuzzywuzzy import fuzz
from paddle.io import Dataset
from sklearn.model_selection import StratifiedKFold

from runconfig import lac2id, dep2id
from utils.log_setting import setlog

logger = setlog.logger
DEBUG = False

# 装载分词模型
lac = LAC(mode='lac')

class RawData(object):
    def __init__(self, ):
        BQ_PATH = '../raw_data/BQ/'
        self.bq_dev = pd.read_csv(os.path.join(BQ_PATH, 'dev'), sep='\t', error_bad_lines=False, header=None,
                             names=['text_a', 'text_b', 'label'])
        self.bq_test = pd.read_csv(os.path.join(BQ_PATH, 'test'), sep='\t', error_bad_lines=False, header=None,
                              names=['text_a', 'text_b', 'label'])
        self.bq_train = pd.read_csv(os.path.join(BQ_PATH, 'train'), sep='\t', error_bad_lines=False, header=None,)
        self.bq_train.columns =  ['text_a','text_b','label']

        self.bq_dev['domain'] = 0
        self.bq_train['domain'] = 0
        self.bq_test['domain'] = 0

        LCQMC_PATH = '../raw_data/LCQMC/'
        self.lcqmc_dev = pd.read_csv(os.path.join(LCQMC_PATH, 'dev'), sep='\t', error_bad_lines=False, header=None,
                                names=['text_a', 'text_b', 'label'])
        self.lcqmc_test = pd.read_csv(os.path.join(LCQMC_PATH, 'test'), sep='\t', error_bad_lines=False, header=None,
                                 names=['text_a', 'text_b', 'label'])
        self.lcqmc_train = pd.read_csv(os.path.join(LCQMC_PATH, 'train'), sep='\t', error_bad_lines=False, header=None,
                                  names=['text_a', 'text_b', 'label'])
        self.lcqmc_dev['domain'] = 1
        self.lcqmc_test['domain'] = 1
        self.lcqmc_train['domain'] = 1

        OPPO_PATH = '../raw_data/OPPO'
        self.oppo_dev = pd.read_csv(os.path.join(OPPO_PATH, 'dev'), sep='\t', error_bad_lines=False, header=None,
                               names=['text_a', 'text_b', 'label'])
        self.oppo_train = pd.read_csv(os.path.join(OPPO_PATH, 'train'), sep='\t', error_bad_lines=False, header=None,
                                 names=['text_a', 'text_b', 'label'])
        self.oppo_extra = pd.read_csv(os.path.join(OPPO_PATH,'../../code1/data_new/gaiic_track3_round1_train_20210220.tsv'), sep='\t', error_bad_lines=False, header=None,
                               names=['text_a', 'text_b', 'label'])
        self.oppo_dev['domain'] = 2
        self.oppo_train['domain'] = 2
        self.oppo_extra['domain'] = 2

        self.test_data = pd.read_csv('../raw_data/test_A.tsv', sep= '\t', header=None)
        self.test_data.columns = ['text_a', 'text_b']


        self.train_data = pd.concat(
            [self.bq_train, self.bq_test,
             self.lcqmc_test, self.lcqmc_train, self.oppo_train,self.oppo_extra]).dropna().reset_index(drop=True)
        self.dev_data = pd.concat([self.bq_dev, self.lcqmc_dev, self.oppo_dev]).dropna().reset_index(drop=True)

    def stratifiedkfold(self,k):
        kf  = StratifiedKFold(n_splits=k,random_state=1,shuffle=True)
        bq_df = pd.concat([self.bq_train,self.bq_test,self.bq_dev]).dropna().reset_index(drop=True)
        lcqmc_df = pd.concat([self.lcqmc_train,self.lcqmc_test,self.lcqmc_dev]).dropna().reset_index(drop=True)
        oppo_df = pd.concat([self.oppo_train,self.oppo_dev]).dropna().reset_index(drop=True)

        dfs = [bq_df,lcqmc_df,oppo_df]
        indexes = []
        for df in dfs:
            split_index = [(train_index,dev_index) for train_index,dev_index in kf.split(df.index,df['label'])]
            indexes.append(split_index)

        split_datas = []
        for i in range(k):
            train_df = []
            dev_df = []
            for ind,df in zip(indexes,dfs):
                train_index,dev_index = ind[i]
                train_df.append(df.iloc[train_index])
                dev_df.append(df.iloc[dev_index])
            train_df = pd.concat(train_df,axis=0)
            dev_df = pd.concat(dev_df,axis=0)
            split_datas.append((train_df,dev_df))
        return split_datas

        pass
    def kfold(self):
        pass

    def getTrain(self,debug = False):

        return self.train_data if not debug else self.bq_train

    def getDev(self,debug = False):

        return self.dev_data if not debug else self.bq_dev

    def getTest(self):

        return self.test_data

    def __getitem__(self, item):
        pass


class RawDataNew(object):
    def __init__(self, ):
        BQ_PATH = './data_new/BQ/'
        self.bq_dev = pd.read_csv(os.path.join(BQ_PATH, 'dev'), sep='\t', error_bad_lines=False,
                                  )
        self.bq_dev['domain'] = 0
        self.bq_test = pd.read_csv(os.path.join(BQ_PATH, 'test'), sep='\t', error_bad_lines=False,
                                   )
        self.bq_test['domain'] = 0
        self.bq_train = pd.read_csv(os.path.join(BQ_PATH, 'train'), sep='\t', error_bad_lines=False, )
        self.bq_train['domain'] = 0

        LCQMC_PATH = './data_new/LCQMC/'
        self.lcqmc_dev = pd.read_csv(os.path.join(LCQMC_PATH, 'dev'), sep='\t', error_bad_lines=False,
                                     )
        self.lcqmc_dev['domain'] = 1
        self.lcqmc_test = pd.read_csv(os.path.join(LCQMC_PATH, 'test'), sep='\t', error_bad_lines=False,
                                      )
        self.lcqmc_test['domain'] = 1
        self.lcqmc_train = pd.read_csv(os.path.join(LCQMC_PATH, 'train'), sep='\t', error_bad_lines=False,
                                       )
        self.lcqmc_train['domain'] = 1

        OPPO_PATH = './data_new/OPPO'
        self.oppo_dev = pd.read_csv(os.path.join(OPPO_PATH, 'dev'), sep='\t', error_bad_lines=False, )
        self.oppo_dev['domain'] = 2
        self.oppo_train = pd.read_csv(os.path.join(OPPO_PATH, 'train'), sep='\t', error_bad_lines=False,)
        self.oppo_train['domain'] = 2
        self.oppo_extra = pd.read_csv(os.path.join(OPPO_PATH, 'gaiic_track3_round1_train_20210220.tsv'), sep='\t',
                                      error_bad_lines=False,
                                      )
        self.oppo_extra['domain'] = 2
        # self.oppo_extra2 = pd.read_csv(os.path.join(OPPO_PATH,'gaiic_track3_round2_train_2021040.tsv'), sep='\t', error_bad_lines=False, header=None,
        #                        names=['text_a', 'text_b', 'label'])
        self.test_data = pd.read_csv('../raw_data/test_A.tsv', sep= '\t', header=None)
        self.test_data.columns = ['text_a', 'text_b']

        # use_all
        self.train_data = pd.concat(
            [self.bq_train, self.bq_test,
             self.lcqmc_test, self.lcqmc_train, self.oppo_train, self.oppo_extra]).dropna().reset_index(drop=True)
        self.dev_data = pd.concat([self.bq_dev, self.lcqmc_dev, self.oppo_dev]).dropna().reset_index(drop=True)

    def stratifiedkfold(self,k):
        kf  = StratifiedKFold(n_splits=k,random_state=2,shuffle=True)
        bq_df = pd.concat([self.bq_train,self.bq_test,self.bq_dev]).dropna().reset_index(drop=True)
        lcqmc_df = pd.concat([self.lcqmc_train,self.lcqmc_test,self.lcqmc_dev]).dropna().reset_index(drop=True)
        oppo_df = pd.concat([self.oppo_train,self.oppo_dev]).dropna().reset_index(drop=True)

        dfs = [bq_df,lcqmc_df,oppo_df]
        indexes = []
        for df in dfs:
            split_index = [(train_index,dev_index) for train_index,dev_index in kf.split(df.index,df['label'])]
            indexes.append(split_index)

        split_datas = []
        for i in range(k):
            train_df = []
            dev_df = []
            for ind,df in zip(indexes,dfs):
                train_index,dev_index = ind[i]
                train_df.append(df.iloc[train_index])
                dev_df.append(df.iloc[dev_index])
            train_df = pd.concat(train_df,axis=0)
            dev_df = pd.concat(dev_df,axis=0)
            split_datas.append((train_df,dev_df))
        return split_datas

        pass
    def kfold(self):
        pass

    def getTrain(self,debug = False):

        return self.train_data.sample(1000) if debug else self.train_data

    def getDev(self,debug = False):

        return self.dev_data.sample(1000) if debug else self.dev_data

    def getTest(self):

        return self.test_data

    def __getitem__(self, item):
        pass

class QMSet(Dataset):
    '''
    提取一些特征、以及将每行数据转换为json形式
    '''
    def __init__(self,df,choice=1):

        if choice==1:
            print('init data set ,lac feat!')
            for col in ['deprel_a', 'deprel_b', 'postag_a', 'postag_b', 'word_a', 'word_b']:
                if isinstance(df[col].iloc[0],str):
                    df[col] = df[col].apply(lambda x:eval(x))
            df['ratio'] = df.apply(lambda x: fuzz.ratio(x.text_a, x.text_b), axis=1) #编辑距离
            self.data = df
        elif choice==2:
            print('init data set origin feat!')
            df['text_a'] = df['text_a'].apply(removeOral)
            df['text_b'] = df['text_b'].apply(removeOral)
            df['ratio'] = df.apply(lambda x: fuzz.ratio(x.text_a, x.text_b), axis=1)  # 编辑距离
            self.data = df
        elif choice ==3:
            print('init data set origin feat! , select ratio > 70')
            df['text_a'] = df['text_a'].apply(removeOral)
            df['text_b'] = df['text_b'].apply(removeOral)
            df['ratio'] = df.apply(lambda x: fuzz.ratio(x.text_a, x.text_b), axis=1)  # 编辑距离
            df = df[df.ratio > 70]
            self.data = df
        elif choice ==4:
            print('init data set origin feat! , select ratio <= 70')
            df['text_a'] = df['text_a'].apply(removeOral)
            df['text_b'] = df['text_b'].apply(removeOral)
            df['ratio'] = df.apply(lambda x: fuzz.ratio(x.text_a, x.text_b), axis=1)  # 编辑距离
            df = df[df.ratio <= 70]
            self.data = df
    def __getitem__(self, item):
        '''

        :param item:返回第item个元素的json 字典形式
        :return:
        '''
        sample = self.data.iloc[item].to_dict()
        return sample
    def __len__(self,):
        return self.data.shape[0]

def getMaskIndexWithLac(example ,all_select):
    '''
    使用百度的lac分词器
    :param x1:
    :param x2:
    :return:
    '''
    #分词
    x1 = example['text_a']
    x2 = example['text_b']

    lac1 = example['postag_a']
    lac2 = example['postag_b']

    s1 = example['word_a']
    s2 = example['word_b']

    deprel1 = example['deprel_a']
    deprel2 = example['deprel_b']


    if all_select: #全部为1
        index1 = np.ones((len(x1),800))
        index2 = np.ones((len(x2),800))
    else: #
        index1 = np.zeros((len(x1), 800))
        index2 = np.zeros((len(x2), 800))

    lac12id = np.zeros(len(x1))
    lac22id = np.zeros(len(x2))

    dep2id1 = np.zeros(len(x1))
    dep2id2 = np.zeros(len(x2))

    assert len(s1) == len(lac1) and len(s1) == len(deprel1),'长度不一致！%s _ %s _ %s' % (len(s1),len(lac1),len(deprel1))
    assert len(s2) == len(lac2) and len(s2) == len(deprel2), '长度不一致！！%s _ %s _ %s'% (len(s2),len(lac2),len(deprel2))
    i = 0
    for w,l,d in  zip(s1,lac1,deprel1):
        if w not in s2 and not all_select:
            for j in range(i,i+len(w)):
                if j < len(x1):
                    index1[j,:] = 1
        for j in range(i,i+len(w)):
            if j==i and j < len(x1):
                dep2id1[j] = dep2id.get('B-' + d, 0)
                lac12id[j] = lac2id.get('B-' + l, 0)
            elif j < len(x1):
                dep2id1[j] = dep2id.get('I-' + d, 0)
                lac12id[j] = lac2id.get('I-' + l, 0)
            else:
                break
        i += len(w)

    i = 0
    for w,l,d in zip(s2,lac2,deprel2):
        if w not in s1 and not  all_select:
            for j in range(i,i+len(w)):
                if j<len(x2):
                    index2[j,:] = 1
        for j in range(i,i+len(w)):
            if j==i and j < len(x2):
                dep2id2[j] = dep2id.get('B-' + d, 0)
                lac22id[j] = lac2id.get('B-' + l, 0)
            elif j < len(x2):
                dep2id2[j] = dep2id.get('I-' + d, 0)
                lac22id[j] = lac2id.get('I-' + l, 0)
            else:
                break
        i += len(w)

    return index1,index2,lac12id,lac22id,dep2id1,dep2id2

def getMaskIndex(x1,x2):
    '''
    jieba 分词器
    :param x1:
    :param x2:
    :return:
    '''
    index1 = np.zeros((len(x1),768))
    index2 = np.zeros((len(x2),768))
    s1,s2  = list(jieba.cut(x1)),list(jieba.cut(x2))

    s1 = [(i,v) for i, v in enumerate(s1)]
    s2 = [(i,v) for i, v in enumerate(s2)]

    i = 0
    for n,w in enumerate(s1):
        w = w[1]
        if (n,w) not in s2:
            for j in range(i,i+len(w)):
                index1[j,:] = 1
        i += len(w)
    i = 0
    for n,w in enumerate(s2):
        w = w[1]
        if (n,w) not in s1:
            for j in range(i,i+len(w)):
                index2[j,:] = 1
        i += len(w)
    return index1,index2

def removeOral(string):
    startslist = ['有谁知道','大家知道','你知道','谁知道','谁了解','有谁了解','大家了解','你了解']
    endslist = ['吗']
    s,e = False,False
    for st in startslist:
        if string.startswith(st):
            s = True
            break
    for end in endslist:
        if string.endswith(end):
            e = True
            break
    if s and e:
        tmp = re.findall(st + '(.*)' + end,string)[0]
        if len(tmp)==0:
            return 'None'
        else:
            return tmp
    return string

def read_text_pair_base(data_path, is_test=False):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}


def markSentenceWithDiff(s1, s2,ratio ):
    if fuzz.ratio(s1, s2) < ratio:
        return s1, s2
    s1, s2 = list(jieba.cut(s1)), list(jieba.cut(s2))
    s1 = [(i,v) for i ,v in enumerate(s1)]
    s2 = [(i,v) for i,v in enumerate(s2)]

    new_s1, new_s2 = '', ''
    for i,w in enumerate(s1):
        if (i,w) in s2:
            new_s1 += w
        else:
            tmp = '@' + w + '@'
            new_s1 += tmp

    for i,w in enumerate(s2):
        if (i,w) in s1:
            new_s2 += w
        else:
            tmp = '#' + w + '#'
            new_s2 += tmp
    return new_s1, new_s2

def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True,
        )

def convert_example(example, tokenizer,ratio, max_seq_length=512, is_test=False):

    query, title = example["text_a"], example["text_b"]
    # query,title = markSentenceWithDiff(query,title,ratio)
    title = title +'[SEP]'+example['operation']

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def convert_example_with_attention_token(example, tokenizer, ratio,max_seq_length=512, is_test=False):
    '''

    :param example:
    :param tokenizer:
    :param max_seq_length:
    :param is_test:
    :return:
    '''
    TOKEN_MASK_SHAPE = (1,768)
    query, title = example["text_a"], example["text_b"]
    max_l = (max_seq_length - 3) // 2

    if len(query) > max_l:
        query = query[:max_l]
    if len(title) > max_l:
        title = title[:max_l]

    # operation = example['operation'][:40]
    # operation_l = len(operation)

    #对偶
    if np.random.rand() > 0.5 and not is_test:
        query,title = title,query

    if example['ratio'] >ratio:
        ind1, ind2 = getMaskIndex(query, title)
    else:
        ind1 = np.ones((len(query),768))
        ind2 = np.ones((len(title),768))

    input_tokens = ['[CLS]'] + [c for c in query] + ['[SEP]'] + [c for c in title] + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    token_type_ids = [0] * (len(query) + 2) + [1] * (len(title) + 1)
    sequence_length = len(input_ids)
    assert  len(input_ids) == len(token_type_ids)

    select_index = np.concatenate([np.ones(TOKEN_MASK_SHAPE),ind1,np.ones(TOKEN_MASK_SHAPE),ind2,np.ones(TOKEN_MASK_SHAPE)])
    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, select_index,label
    else:
        return input_ids, token_type_ids, select_index

def convert_example_with_attention_token_domain(example, tokenizer, ratio,max_seq_length=512, is_test=False):
    '''

    :param example:
    :param tokenizer:
    :param max_seq_length:
    :param is_test:
    :return:
    '''
    TOKEN_MASK_SHAPE = (1,768)
    query, title = example["text_a"], example["text_b"]
    max_l = (max_seq_length - 3) // 2

    if len(query) > max_l:
        query = query[:max_l]
    if len(title) > max_l:
        title = title[:max_l]

    # operation = example['operation'][:40]
    # operation_l = len(operation)

    #对偶
    if np.random.rand() > 0.5 and not is_test:
        query,title = title,query

    if example['ratio'] >ratio:
        ind1, ind2 = getMaskIndex(query, title)
    else:
        ind1 = np.ones((len(query),768))
        ind2 = np.ones((len(title),768))

    input_tokens = ['[CLS]'] + [c for c in query] + ['[SEP]'] + [c for c in title] + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    token_type_ids = [0] * (len(query) + 2) + [1] * (len(title) + 1)
    sequence_length = len(input_ids)
    assert  len(input_ids) == len(token_type_ids)

    select_index = np.concatenate([np.ones(TOKEN_MASK_SHAPE),ind1,np.ones(TOKEN_MASK_SHAPE),ind2,np.ones(TOKEN_MASK_SHAPE)])
    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        domain = example['domain']
        return input_ids, token_type_ids, select_index,domain,label
    else:
        return input_ids, token_type_ids, select_index

def convert_example_with_lac(example, tokenizer, ratio,max_seq_length=512, is_test=False):
    TOKEN_MASK_SHAPE = (1,800)
    query, title = example["text_a"], example["text_b"]

    len_query,len_title = len(query),len(title)
    if max_seq_length - 3 < len_query + len_title: #超过长度
        over_size = len_query + len_title - max_seq_length + 3 #超了多少长度
        l = (over_size + 1) // 2
        query = query[:l]
        title = title[:l]
        example['text_a'] = query
        example['text_b'] = title
        warnings.warn("data was cutted!")


    ind1, ind2, lac12id, lac22id, dep2id1, dep2id2 = getMaskIndexWithLac(example,example['ratio'] < ratio)

    input_tokens = ['[CLS]'] + [c for c in query] + ['[SEP]'] + [c for c in title] + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    token_type_ids = [0] * (len(query) + 2) + [1] * (len(title) + 1)
    dep_feat = np.concatenate([np.array([0]) , dep2id1 , np.array([0]) , dep2id2 , np.array([0])])
    lac_feat = np.concatenate([np.array([0]) , lac12id , np.array([0]) , lac22id , np.array([0])]) #
    sequence_length = len(input_ids)
    assert  len(input_ids) == len(token_type_ids)

    select_index = np.concatenate([np.ones(TOKEN_MASK_SHAPE),ind1,np.ones(TOKEN_MASK_SHAPE),ind2,np.ones(TOKEN_MASK_SHAPE)])
    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, select_index,lac_feat,dep_feat,sequence_length,label
    else:
        return input_ids, token_type_ids, select_index,lac_feat,dep_feat,sequence_length




if __name__ == "__main__":

    #训前的配置


    rd = RawData()
    data = rd.stratifiedkfold(5)