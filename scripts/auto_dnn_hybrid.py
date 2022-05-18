# -*- coding:utf-8 -*-

import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import read_batch
import auc

class AutoHybridModel(nn.Module):
    def __init__(self):
        super(AutoHybridModel, self).__init__()
        self.threads  = None
        self.needinit = True
        self.cuda   = 'cuda'
        self.device =  torch.device(self.cuda if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print('torch.cuda: ', torch.cuda.device_count(), torch.cuda.current_device())
        else:
            print('GPU is not available!!!')

        self.train_path = r'../data/movielens/train_data.csv'
        self.test_path  = r'../data/movielens/test_data.csv'

        self.emb_size       = 16
        self.single_size    = 3
        self.multi_size     = 4
        self.mask_size      = 4
        self.nn_size        = [64, 32, 1]
        self.batch_size     = 512
        self.learning_rate  = [0.005, 1e-2]
        self.epoch          = 2
        self.t_batch        = 10
        self.global_step    = 0

        self.pooling_module = ["sum", "mean", "max", "min", "korder", "atten"]
        self.fea_size       = {"userId": 139000, "movieId": 132000, "tagId": 41000, "genreId": 30, 'year': 150}
        self.single_name    = ["userId", "movieId", "year"]
        self.multi_name     = {"user_genre": "genreId", "urb": "movieId", "movie_genre": "genreId", "movie_tag": "tagId"}
        # modules:
        self.Emb_layers = nn.ModuleDict({name: nn.Embedding(self.fea_size[name], self.emb_size) for name in self.fea_size})
        self.layer1     = nn.Linear(((self.single_size+self.multi_size)*self.emb_size), self.nn_size[0])
        self.layer2     = nn.Linear(self.nn_size[0], self.nn_size[1])
        self.layer3     = nn.Linear(self.nn_size[1], self.nn_size[2])
        self.sumBN      = nn.ModuleDict({name: nn.BatchNorm1d(self.emb_size) for name in self.multi_name})
        self.meanBN     = nn.ModuleDict({name: nn.BatchNorm1d(self.emb_size) for name in self.multi_name})
        self.maxBN      = nn.ModuleDict({name: nn.BatchNorm1d(self.emb_size) for name in self.multi_name})
        self.minBN      = nn.ModuleDict({name: nn.BatchNorm1d(self.emb_size) for name in self.multi_name})
        self.attenBN    = nn.ModuleDict({name: nn.BatchNorm1d(self.emb_size) for name in self.multi_name})
        self.korderBN       = nn.ModuleDict({name: nn.BatchNorm1d(self.emb_size) for name in self.multi_name})

        self.alpha = {
            name: torch.FloatTensor([1] * len(self.pooling_module)).requires_grad_() for name in self.multi_name
            }
        self.atten_size    = {"movieId": 132000, "tagId": 41000, "genreId": 30}
        self.atten_layers  = nn.ModuleDict({name: nn.Embedding(self.atten_size[name], 1) for name in self.atten_size})

    def pooling_fun(self, masked_emb, alpha, name, atten_w=None):
        """ 
        pooling_module : {sum, mean, max, min, atten, korder}
        masked_emb     : [b, max_fea_num, emb_size]
        atten_w        : [b, max_fea_num, 1]
        narmal_fum     : layer normalization on embedding dim
        alpha          : structure parameters in [6,]
        """
        emb_size  = masked_emb.shape[-1]
        sum_pool  = torch.sum(masked_emb, dim=1, keepdim=True)         # [b, 1, emb_size]
        sum_pool  = self.sumBN[name](sum_pool.view([-1, emb_size])).view([-1, 1, emb_size])
        mean_pool = torch.mean(masked_emb, dim=1, keepdim=True)         # [b, 1, emb_size]
        mean_pool = self.meanBN[name](mean_pool.view([-1, emb_size])).view([-1, 1, emb_size])

        maxL2_emb = torch.sum(torch.square(masked_emb), dim=-1)
        minL2_emb = torch.sum(torch.square(masked_emb), dim=-1)
        max_index = torch.max(maxL2_emb, dim=1)[1]
        minL2_emb = torch.where(minL2_emb == 0, torch.ones_like(minL2_emb).to(self.device)*9999, minL2_emb)
        min_index = torch.min(minL2_emb, dim=1)[1]
        max_mask  = torch.zeros(masked_emb.shape[1], masked_emb.shape[0]).to(self.device).scatter(0, max_index.view(1,-1), 1)
        max_mask  = max_mask.transpose(1, 0).view(masked_emb.shape[0], masked_emb.shape[1], 1)
        min_mask  = torch.zeros(masked_emb.shape[1], masked_emb.shape[0]).to(self.device).scatter(0, min_index.view(1,-1), 1)
        min_mask  = min_mask.transpose(1, 0).view(masked_emb.shape[0], masked_emb.shape[1], 1)
        
        max_pool   = torch.sum(masked_emb * max_mask, dim=1, keepdim=True)           # [b, 1, emb_size]
        max_pool   = self.maxBN[name](max_pool.view([-1, emb_size])).view([-1, 1, emb_size])
        min_pool   = torch.sum(masked_emb * min_mask, dim=1, keepdim=True)           # [b, 1, emb_size]
        min_pool   = self.minBN[name](min_pool.view([-1, emb_size])).view([-1, 1, emb_size])
        atten_w    = torch.softmax(atten_w, dim=1)
        atten_pool = torch.sum(torch.mul(masked_emb, atten_w), dim=1, keepdim=True)  # [b, 1, emb_size]
        atten_pool = self.attenBN[name](atten_pool.view([-1, emb_size])).view([-1, 1, emb_size])
        
        squared_sum  = torch.square(torch.sum(masked_emb, dim=1, keepdim=True))    # [b, 1, emb_size]
        sumed_square = torch.sum(torch.square(masked_emb), dim=1, keepdim=True)    # [b, 1, emb_size]
        korder_pool      = nn.functional.normalize(0.5 * (squared_sum - sumed_square), p=2, dim=-1)      # [b, 1, emb_size]
        korder_pool      = self.korderBN[name](korder_pool.view([-1, emb_size])).view([-1, 1, emb_size])
        
        normaled_pool = torch.cat([
            sum_pool, mean_pool, max_pool, min_pool, korder_pool, atten_pool
            ], dim=1).view([-1, len(self.pooling_module), emb_size])           # [b, 6, emb_size]
        
        weight = torch.softmax(alpha.to(self.device), dim=0).view([1, len(self.pooling_module), 1])    # [1, 6, 1]
        weighted_pool = torch.sum(torch.mul(normaled_pool, weight), dim=1, keepdim=False)  # [b, emb_size]
        return weighted_pool

    def mask_fun(self, emb_fea, mask_fea):
        """
        emb_fea  : [b, max_fea_size, emb_size]
        mask_fea : [b, 1]
        """
        max_len    = emb_fea.shape[1]
        mask       = mask_fea.view([-1, 1]).expand([-1, max_len])      # [b, max_fea_size]
        mask_range = torch.arange(0, max_len).to(self.device).expand_as(mask)          # [b, max_fea_size]
        the_mask   = (mask > mask_range).unsqueeze(-1)                # [b, max_fea_size, 1]
        emb_fea    = torch.where(the_mask, emb_fea, torch.zeros_like(emb_fea).to(self.device))          # [b, max_fea_size, emb_size]
        return emb_fea

    def forward(self, single_fea, multi_fea, mask_fea):
        """
        single_fea : ["userId", "movieId", "year"]
        multi_fea  : ["user_genre", "urb", "movie_genre", "muvie_tag"]
        mask_fea   : ["user_genre_num", "urb_num", "movie_genre_num", "muvie_tag_num"]     
        """
        # embedding part
        emb1 = torch.cat([
            self.Emb_layers[name](feature) for name, feature in zip(self.single_name, single_fea)
            ], dim=1)       # [b, 3 * emb_size]

        emb2  = []
        for name, feature, mask in zip(self.multi_name, multi_fea, mask_fea):
            masked_emb = self.mask_fun(self.Emb_layers[self.multi_name[name]](feature), mask)       # [b, max_fea_num, emb_size]
            atten_w    = self.mask_fun(self.atten_layers[self.multi_name[name]](feature), mask)     # [b, max_fea_num, 1]
            pooled_emb = self.pooling_fun(
                masked_emb, alpha=self.alpha[name], name=name, atten_w=atten_w
                )     # [b, emb_size]
            emb2.append(pooled_emb)
        emb2 = torch.cat(emb2, dim=1)       # [b, 4 * emb_size]

        # nn part
        totoal_fea = torch.cat([emb1, emb2], dim=1)      # [b, (3+4)*emb_size]
        output = torch.relu(self.layer1(totoal_fea))                # [b, 128]
        output = torch.relu(self.layer2(output))                    # [b, 64]
        output = torch.sigmoid(self.layer3(output))                 # [b, 1]
        output    = output.view([-1])       # [b, ]
        return output

    def chuck_reader(self, path):
        data = pd.read_csv(path, chunksize=self.batch_size, sep=';')
        return data

    def init_param(self):
        print('initialize parameters...')
        model = self.train()
        for param in model.atten_layers.parameters():
            param.data = torch.FloatTensor(param.data.size()).normal_(1., 1e-6).to(self.device)

    def fit(self):
        if self.threads:
            torch.set_num_threads(self.threads)
        print('THREADS: ',torch.get_num_threads())
        if self.needinit:
            self.init_param()
        model = self.train()
        if torch.cuda.is_available():
            model.to(self.device)
        
        criterion  = nn.BCELoss(reduction='mean')
        optimizer1 = torch.optim.Adam(model.parameters(), lr = self.learning_rate[0], weight_decay = 1e-6)
        optimizer2 = torch.optim.Adam(model.alpha.values(), lr = self.learning_rate[1], weight_decay = 1e-6)
        print("===== Model Structure =====")
        print(model)

        for epoch in range(self.epoch):
            print('===== epoch: {:d} ====='.format(epoch+1))
            data  = self.chuck_reader(self.train_path)
            start = time.time()
            slide_loss = 0
            slide_auc  = 0
            for i, df in enumerate(data):
                self.global_step += 1
                label, single_val, mulit_val, mask_val = read_batch.data_split(df, self.cuda)
                label = label.view([-1])
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                predict = model(single_val, mulit_val, mask_val)
                loss    = criterion(predict, label)
                slide_loss += loss
                the_label = torch.cat([predict.view([-1,1]), label.view([-1,1])], dim = 1)
                if torch.cuda.is_available():
                    the_label = the_label.detach().cpu().numpy().tolist()
                else:
                    the_label = the_label.detach().numpy().tolist()
                
                loss.backward()
                if (i+1) % self.t_batch == 0:
                    optimizer2.step()
                else:
                    optimizer1.step()

                slide_auc += auc.auc(the_label)
                if (i+1) % 100 == 0:
                    end = time.time()
                    minute  = int((end - start) // 60)
                    seconds = (end - start) % 60
                    print('----- batch: {:d} loss: {:.4f} AUC: {:.4f} use: {:d}min {:.2f}s -----'.format(i+1, slide_loss.data.item()/(i+1), slide_auc/(i+1), minute, seconds))
                    print('----- Structure parameters: -----')
                    for name in model.alpha:
                        print(name, ':', model.alpha[name])

    def eval_by_batch(self):
        if self.threads:
            torch.set_num_threads(self.threads)
        print('THREADS: ',torch.get_num_threads())
        with torch.no_grad():
            model     = self.eval()
            if torch.cuda.is_available():
                model.to(self.device)
            criterion = nn.BCELoss(reduction='mean')
            result    = torch.Tensor()          # CPU tensor
            loss      = 0 
            slide_auc = 0 
            data      = self.chuck_reader(self.test_path)
            start     = time.time()
            for i, df in enumerate(data):
                label, single_val, mulit_val, mask_val = read_batch.data_split(df, self.cuda)
                label = label.view([-1])

                predict   = model(single_val, mulit_val, mask_val)
                loss      += criterion(predict, label)
                predict2  = torch.where(predict > 0.5, torch.ones_like(predict), torch.zeros_like(predict))
        
                the_label = torch.cat([predict.view([-1,1]), label.view([-1,1])], dim = 1).detach().cpu()       # move to CPU
                result    = torch.cat([result, the_label], dim = 0)
                the_label = the_label.numpy().tolist()
                slide_auc += auc.auc(the_label)
                if (i+1) % 50 == 0:
                    end = time.time()
                    minute  = int((end - start) // 60) 
                    seconds = (end - start) % 60
                    print('----- batch: {:d} loss: {:.4f} AUC: {:.4f} use: {:d}min {:.2f}s -----'.format(i+1, loss.data.item()/(i+1), slide_auc/(i+1), minute, seconds))

        return result, loss / (i+1)

if __name__ == "__main__":
    # torch.manual_seed(2022)
    themodel = AutoHybridModel()
    print('===== train stage =====')
    start = time.time()
    train = themodel.fit()
    end   = time.time()
    print('===== train stage finished in {:d}min {:d}s ====='.format(int((end-start) // 60), int((end-start) % 60)))

    print('===== test session =====')
    start        = time.time()
    result, loss = themodel.eval_by_batch()
    test_label   = result.detach().numpy().tolist()
    end = time.time()
    print("===== loss in test set is: {:.4f} =====\n===== AUC in test set is: {:.4f} =====".format(loss.data.item(), auc.auc(test_label)))
    print('===== test stage finished in {:d}min {:d}s ====='.format(int((end-start) // 60), int((end-start) % 60)))
