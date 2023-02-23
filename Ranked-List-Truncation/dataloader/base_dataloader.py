import pickle
from collections import defaultdict
import re

from numpy.core.function_base import _linspace_dispatcher
import torch as t
import numpy as np
from torch.utils import data
import os
import json
# from utils import *
# np.set_printoptions(precision=20)
t.set_printoptions(precision=20)
def cos(x,y):
    assert len(x) == len(y)
    xy=0.0
    x2=0.0
    y2=0.0
    for i in range(len(x)):
        xy += x[i]*y[i]   
        x2 += x[i]**2     
        y2 += y[i]**2     
    return xy/((x2*y2)**0.5)

def process_embd(path):
    raw_lines = open(path, 'r').readlines()
    lines = [eval(line) for line in raw_lines]
    # w_dic = defaultdict(lambda: defaultdict (lambda: 0.0))
    w_dic = {}
    for dic in lines:
        qid, cid = dic['id_'].split('_')
        if qid not in w_dic:
            w_dic[qid] = {cid:{'embd':dic['embd'], 'cembd':dic['cembd']}}
        else:
            w_dic[qid][cid] = {'embd':dic['embd'], 'cembd':dic['cembd']}
    return w_dic


def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)
 
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
 
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

class Rank_Dataset(data.Dataset):
    def __init__(self, model='attncut', retrieve_data: str='lecard', dataset_name: str='slr_test', iter_num=1):
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_prepare(model, retrieve_data, dataset_name, iter_num)
    
    def neighbor_feature(self, dic, key2s, name):
        if name == 'neighbor_tfidf':
            # neighbor_list = [dic[key2s[i]][name][key2s[i-1]] for i in range(1, len(key2s))]
            neighbor_list = [[dic[key2s[i]][name][key2s[i-1]], dic[key2s[i]][name][key2s[i+1]]] for i in range(1, len(key2s)-1)]
            return np.array([[1, dic[key2s[0]][name][key2s[1]]]] + neighbor_list + [[dic[key2s[-1]][name][key2s[-2]],0]])
        elif name == 'cembd':
            neighbor_list = [[cos(dic[key2s[i]][name], dic[key2s[i-1]][name]), cos(dic[key2s[i]][name], dic[key2s[i+1]][name])] for i in range(1, len(key2s)-1)]
            return np.array([[1, cos(dic[key2s[0]][name], dic[key2s[1]][name])]] + neighbor_list + [[cos(dic[key2s[-1]][name], dic[key2s[-2]][name]),0]])

    def data_ready(self, model, data_raw, stats, gt, retrieve_data):
        x, y = [], []
        for key in data_raw:
            scores = np.array(list(data_raw[key].values()))
            key2s = list(data_raw[key].keys())
            # print(scores)
            if model == 'attncut':
                doc_lens = np.array([stats[key][key2]['doclen'] for key2 in data_raw[key]])
                neighbor_tfidf = self.neighbor_feature(stats[key], key2s, 'neighbor_tfidf')
                input_features = np.column_stack((scores, doc_lens, neighbor_tfidf))

            elif model == 'bicut':
                doc_lens = np.array([stats[key][key2]['doclen'] for key2 in data_raw[key]])
                input_features = np.column_stack((scores, doc_lens))
            elif model == 'lecut':
                doc_lens = np.array([stats[key][key2]['doclen'] for key2 in data_raw[key]])
                neighbor_tfidf = self.neighbor_feature(stats[key], key2s, 'neighbor_tfidf')
                # neighbor_embd = self.neighbor_feature(stats[key], key2s, 'cembd')
                embd = np.array([stats[key][key2]['embd'] for key2 in data_raw[key]])
                # cos_embd = np.array([stats[key][key2]['cos_embd'] for key2 in data_raw[key]])

                cembd = embd#[stats[key][key2]['cembd'] for key2 in data_raw[key]]
                
                cos_avg_cembd = [1] + [cos(cembd[i], np.sum(softmax(scores[:i], -1)[0].reshape(-1,1)*cembd[:i],0)) for i in range(1, len(key2s))]
                # print(softmax(scores[:30], -1)[0])

                input_features = np.column_stack((scores, doc_lens, neighbor_tfidf, cos_avg_cembd, embd))
                # input_features = np.array([[score] for score in scores])
            elif model == 'choppy': #choppy
                input_features = scores
            
            if retrieve_data == 'COLIEE2020':
                is_rel = list(map(lambda x: 1 if (x in gt[key] and gt[key][x]==1) else 0, data_raw[key].keys()))
            else:
                is_rel = list(map(lambda x: 1 if (x in gt[key] and gt[key][x]>=2) else 0, data_raw[key].keys()))
            # print(is_rel)
            # if sum(is_rel) == 0: print(key)
            
            x.append(input_features.tolist())
            y.append(is_rel)
        # print(y)
        return x, y

    def to_data_raw(self, lines, type=1): # from neural results to score dics
        if type == 1:
            data_raw = {}
            for line in lines:
                dic = eval(line)
                qid, cid = dic['id_'].split('_')
                if qid not in data_raw:
                    data_raw[qid] = {cid: dic['res'][1] - dic['res'][0]}
                else:# len(list(data_raw[qid].keys())) < 100:
                    data_raw[qid][cid] = dic['res'][1] - dic['res'][0]
        else:
            data_raw = lines
        sorted_data_raw = {}
        for key in data_raw:
            sorted_results = sorted(data_raw[key].items(), key=lambda x: x[1], reverse=True)
            sorted_data_raw[key] = {item[0]:item[1] for item in sorted_results}
        # print({key:sorted_data_raw[key].keys() for key in sorted_data_raw})
        return sorted_data_raw

    def data_prepare(self, model, retrieve_data, dataset_name: str, iter_num):
        # train_data_raw = self.to_data_raw(open('/work/mayixiao/cutoff/preresults/%s/scores/slr_%s_train_top100.json'%(retrieve_data, retrieve_data),'r').readlines())
        # test_data_raw = self.to_data_raw(open('/work/mayixiao/cutoff/preresults/%s/scores/slr_%s_test_top100.json'%(retrieve_data, retrieve_data),'r').readlines())
        train_data_raw = self.to_data_raw(json.load(open('/work/mayixiao/cutoff/results/l2r_%s_%s_train_%i.json'%(retrieve_data, dataset_name, iter_num),'r')),2)
        test_data_raw = self.to_data_raw(json.load(open('/work/mayixiao/cutoff/results/l2r_%s_%s_test_%i.json'%(retrieve_data, dataset_name, iter_num),'r')),2)

        # train_data_raw = self.to_data_raw(open('/work/mayixiao/cutoff/preresults/COLIEE2020/scores/COLIEE2020_roberta_train_top30.json','r'),1)
        # test_data_raw = self.to_data_raw(open('/work/mayixiao/cutoff/preresults/COLIEE2020/scores/COLIEE2020_roberta_test_top30.json','r'),1)
        
        # # write slr.json
        # w_train = {key1: list(train_data_raw[key1].keys()) for key1 in train_data_raw}
        # w_test = {key1: list(test_data_raw[key1].keys()) for key1 in test_data_raw}
        # w_train.update(w_test)
        # json.dump(w_train, open('/work/mayixiao/cutoff/preresults/%s/%s.json'%(retrieve_data, dataset_name),'w'), ensure_ascii=False)
        # print(1/0)
        
        if model in ['attncut', 'bicut']:
            stats = json.load(open('/work/mayixiao/cutoff/preresults/%s/stats.json'%(retrieve_data),'r'))
        elif model == 'choppy':
            stats = -1
        elif model == 'lecut':
            num_map = {'LeCaRD': 100, 'CAIL2021':100, 'COLIEE2020': 30}
            embd_dic = process_embd('/work/mayixiao/cutoff/preresults/%s/embd_%s_%s_top%i.json'%(retrieve_data, retrieve_data, dataset_name, num_map[retrieve_data]))
            stats = json.load(open('/work/mayixiao/cutoff/preresults/%s/stats.json'%(retrieve_data),'r'))
            for key1 in stats:
                for key2 in stats[key1]:
                    if key2 in embd_dic[key1]:
                        stats[key1][key2]['embd'] = embd_dic[key1][key2]['embd']
                        # if retrieve_data == 'COLIEE2020':
                            # stats[key1][key2]['cembd'] = embd_dic[key1][key2]['embd']
                        # else:
                        # stats[key1][key2]['cembd'] = embd_dic[key1][key2]['cembd']
                        # stats[key1][key2]['cos_embd'] = cos(embd_dic[key1][key1]['cembd'], embd_dic[key1][key2]['cembd'])
        if retrieve_data == 'LeCaRD':
            gt = json.load(open('/work/mayixiao/similar_case/LeCaRD/LeCaRD_github/data/label/label_top30_dict.json','r'))
        elif retrieve_data == 'CAIL2021':
            gt = json.load(open('/work/mayixiao/similar_case/202006/data/label/label_top30_dict_2.json','r'))
        elif retrieve_data == 'COLIEE2020':
            gt = json.load(open('/work/mayixiao/cutoff/preresults/COLIEE2020/labels.json','r'))
        
        X_train, y_train = self.data_ready(model, train_data_raw, stats, gt, retrieve_data)
        X_test, y_test = self.data_ready(model, test_data_raw, stats, gt, retrieve_data)

        if model == 'choppy':
            X_train, X_test = t.unsqueeze(t.Tensor(X_train), dim=1).permute(0, 2, 1), t.unsqueeze(t.Tensor(X_test), dim=1).permute(0, 2, 1)
            y_train, y_test = t.Tensor(y_train), t.Tensor(y_test)
        # print(X_train)
        return t.Tensor(X_train), t.Tensor(X_test), t.Tensor(y_train), t.Tensor(y_test)

    def getX_train(self):
        return self.X_train

    def getX_test(self):
        return self.X_test

    def gety_train(self):
        return self.y_train

    def gety_test(self):
        return self.y_test


def dataloader(model='attncut', retrieve_data: str='lecard', dataset_name: str='', batch_size: int=20, iter_num=1):
    """
    batch_ratio: batchsize / datasize
    """
    rank_data = Rank_Dataset(model, retrieve_data, dataset_name, iter_num)

    X_train = rank_data.getX_train()
    X_test = rank_data.getX_test()
    y_train = rank_data.gety_train()
    y_test = rank_data.gety_test()
    
    train_dataset = data.TensorDataset(X_train, y_train)
    test_dataset = data.TensorDataset(X_test, y_test)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_loader, test_loader, rank_data


if __name__ == '__main__':
    a, b, c = dataloader(model='attncut', retrieve_data='LeCaRD', dataset_name='bert',iter_num=1)
    xtr = c.getX_train()
    xte = c.getX_test()
    ytr = c.gety_train()
    yte = c.gety_test()
    print('batch num: ',len(a))
    # print(xtr[0])
  
    pass
