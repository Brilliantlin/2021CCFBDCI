import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import *
def getBestThreshold(y_test, y_pred,thresholds = list(np.arange(0.0, 1.0, 0.001)),disable_tqdm = False):
    '''
    搜索f1分数最好的划分点
    :param y_test:
    :param y_pred:
    :param average: f1_score 的 average
    :param thresholds: 所有需要搜索的阈值
    :param disable_tqdm: 是否使用tqdm
    :return: thresholdOpt,fscoreOpt
    '''


    fscore = np.zeros(shape=(len(thresholds)))
    # print('Length of sequence: {}'.format(len(thresholds)))

    # 拟合模型b
    for index, elem in tqdm(enumerate(thresholds),total=len(thresholds),disable = disable_tqdm ):
        # 修正概率
        y_pred_prob = (y_pred > elem).astype('int')
        # 计算f值
        fscore[index] = accuracy_score(y_test, y_pred_prob)

    # 查找最佳阈值
    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    fscoreOpt = round(fscore[index], ndigits = 4)
    # print('Best Threshold: {} with Acc: {}'.format(thresholdOpt, fscoreOpt))
    return thresholdOpt,fscoreOpt