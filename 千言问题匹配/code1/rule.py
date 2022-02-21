import pandas as pd
import numpy as np
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm.auto import tqdm
from xpinyin import Pinyin
p = Pinyin()
import argparse
def process(train_data):
    '''
    数据处理
    '''
    # 编辑距离
    train_data['ratio'] = train_data.apply(lambda x: fuzz.ratio(x.text_a, x.text_b), axis=1)
    # 包含关系
    # train_data['a_in_b'] = train_data.apply(lambda x: x.text_a in x.text_b, axis=1)  # a 属于 b
    # train_data['b_in_a'] = train_data.apply(lambda x: x.text_b in x.text_a, axis=1)  # b 属于 a
    # 获取拼音
    train_data['text_a_pinyin'] = train_data.apply(lambda x: p.get_pinyin(x.text_a), axis=1)
    train_data['text_b_pinyin'] = train_data.apply(lambda x: p.get_pinyin(x.text_b), axis=1)
    # 获取数学表达式
    # train_data['isMath'] = train_data.apply(lambda x: isMath(x.text_a), axis=1).astype(int)

    return train_data
def rule(test_data):
    '''
    规则
    '''
    cnt = 0
    for i, line in tqdm(test_data.iterrows(), total=test_data.shape[0], desc='规则处理ing'):
        if line['text_a_pinyin'] == line['text_b_pinyin']:
            if line['label'] != 1:
                cnt += 1
                # print(line['text_a'], line['text_b'])
                test_data.loc[i, 'label'] = 1
    print(cnt)
    return test_data
def getRuleTest(test_path,test_sub):
    '''
    获取规则结果
    :param test_path:
    :return:
    '''
    print('🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕 read test data: %s ' % (test_path))
    test_data = pd.read_csv(test_path, sep='\t', header=None)
    test_data.columns = ['text_a', 'text_b']
    test_data['label'] = test_sub
    test_data = process(test_data)
    test_data = rule(test_data)
    return test_data
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, )
    parser.add_argument('--out_file', required=True, )
    args = parser.parse_args()


    test_data = pd.read_csv('../data/test_A.tsv', sep= '\t', header=None)
    test_data.columns = ['text_a', 'text_b']

    test_sub = pd.read_csv(args.input_file, header=None)
    test_data['label'] = test_sub[0]
    test_data = process(test_data)

    test_data = rule(test_data)
    test_data['label'].to_csv(args.out_file, header=None, index=None)
