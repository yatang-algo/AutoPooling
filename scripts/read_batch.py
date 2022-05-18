# -*- coding:utf-8 -*-

import os, sys, time
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")


def data_split(df, cuda = 'cuda'):
    device      = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    label       = np.array(df.rating).reshape(-1, )
    single_val  = [np.array(df[name]) for name in ['userId', 'movieId', 'year']]
    multi_name  = ['user_genre', 'urb', 'genres', 'movie_tag']
    mask_val    = np.array(df[['user_genre_num', 'urb_num', 'genres_len', 'movie_tag_num']])
    multi_val   = []

    length      = mask_val.shape[0]
    width_list  = list(mask_val.max(axis=0))
    for i, name in enumerate(multi_name):
        if width_list[i] == 0:
            multi_val.append(np.zeros([length, 2]))
            break
        if width_list[i] == 1: 
            multi_val.append(np.hstack([np.array(df[name]).reshape([-1,1]), np.zeros([length, 1])]))
            break
        tmp_array = np.zeros([length, width_list[i]])
        for j in range(length):
            multi = list(map(int, df[name].iloc[j].split(',')))
            tmp_array[j,:len(multi)] = multi
        multi_val.append(tmp_array)
    
    mask_val    = [mask_val[:, i] for i in range(mask_val.shape[1])]

    label       = torch.from_numpy(label).to(torch.float32).to(device)
    single_val  = [torch.from_numpy(val).long().to(device) for val in single_val]
    multi_val   = [torch.from_numpy(val).long().to(device) for val in multi_val]
    mask_val    = [torch.from_numpy(val).long().to(device) for val in mask_val]
    return label, single_val, multi_val, mask_val
