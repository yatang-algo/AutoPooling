#coding=utf-8
import os, sys, time
import warnings

warnings.filterwarnings('ignore')

def auc(sample):
    """
    sample: list: [[predict, label],...]
    """
    pos_num = 0
    pos_sum = 0
    if type(sample) is list:
        pass
    else:
        sample = sample.tolist()
    
    sample.sort(key = lambda x: x[0])
    # print(sample)
    for i in range(len(sample)):
        if sample[i][1] == 1:
            pos_num += 1
            pos_sum += i+1
    neg_num = len(sample) - pos_num
    if neg_num == 0 or pos_sum == 0:
        return 1.
    # print(pos_num, pos_sum, neg_num)
    return (pos_sum - (pos_num + 1) * pos_num / 2.) / (neg_num * pos_num)


