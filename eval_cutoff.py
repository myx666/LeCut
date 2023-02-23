# -*- encoding: utf-8 -*-
'''
@Func    :   get ranklist from classification vectors & compute ndcg
@Time    :   2021/07/30 15:43:00
@Author  :   Yixiao Ma 
@Contact :   mayx20@mails.tsinghua.edu.cn
'''

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import math
from scipy import stats
import torch
from eval_vec import eval_metrics, f1, dcg, nci#, init

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--data', type=int, choices=[1,2,3], default=1, help='which dataset')
parser.add_argument('--dataset-name', type=str, default='slr', help='which datasetname')
parser.add_argument('--m', type=str, choices= ['NDCG', 'P', 'MAP', 'KAPPA', 'F1', 'DCG', 'NCI'], default='NDCG', help='Metric.')

args = parser.parse_args()

# id2cos = init(args.data)

def cut_K(dic, K):
    return {key:dic[key][:K] for key in dic}

def to_scores(dic, label_dic):
    return { key:[label_dic[key][str(i)] for i in dic[key]] for key in dic }

def oracle(args, dic, keys, label_dic):
    score_all = 0.0
    
    if args.m == 'F1':
        cut_dic = {} # for case study
        for key in keys:
            leng = len(dic[key])
            pos_scale = 2 if args.data <= 2 else 1
            rels = [i for i in label_dic[key] if label_dic[key][i] >= pos_scale]
            score_all += max([f1(dic[key][:l], rels, label_dic[key], pos_scale) for l in range(1, leng)])
            # for case study
            cut_dic[key] = np.argmax([f1(dic[key][:l], rels, label_dic[key], pos_scale) for l in range(1, leng)])
        print(cut_dic)
    elif args.m == 'DCG':
        for key in keys:
            max_score = 0.0
            leng = len(dic[key])
            for l in range(1,leng):
                ranks = list(map(lambda x: label_dic[key][str(x)] if str(x) in label_dic[key] else 0, dic[key][:l]))
                if sum(ranks) != 0:
                    pos_scale = 2 if args.data <= 2 else 1
                    max_score = max(dcg(ranks, pos_scale), max_score)
            score_all += max_score
        
    elif args.m == 'NCI':
        for key in keys:
            max_score = 0.0
            leng = len(dic[key])
            for l in range(1,leng):
                ranks = list(map(lambda x: label_dic[key][str(x)] if str(x) in label_dic[key] else 0, dic[key][:l]))
                if sum(ranks) != 0:
                    pos_scale = 2 if args.data <= 2 else 1
                    max_score = max(nci(ranks, pos_scale, args.data), max_score)
            score_all += max_score

    return score_all/len(keys)

def greedy(args, keys, keys_train, label_dic, dics):
    leng = 100 if args.data <= 2 else 30
    dics = [cut_K(result,i) for i in range(1,leng+1)]
    all = eval_metrics(args, keys_train, label_dic, dics)
    all2 = [a[0] for a in all]
    grdy_index = all2.index(max(all2)) + 1
    round4(eval_metrics(args, keys, label_dic, [cut_K(result, grdy_index)]))
    print(grdy_index)
    return grdy_index

def sort_dic(data_raw):
    sorted_data = {}
    sorted_results = sorted(data_raw.items(), key=lambda x: x[0])
    sorted_data = {item[0]:item[1] for item in sorted_results}
    return sorted_data

def round4(l):
    for i in l:
        print(round(i[0]*10000)/10000.0)
    return

if args.data == 1:
    with open('/work/mayixiao/similar_case/LeCaRD/LeCaRD_github/data/label/label_top30_dict.json', 'r') as f:
        label_dic = json.load(f)
    keys_train = [i for i in list(label_dic.keys()) if (list(label_dic.keys()).index(i) % 5 != 0 or list(label_dic.keys()).index(i)>=100)]
    keys = [i for i in list(label_dic.keys())[:100] if list(label_dic.keys())[:100].index(i) % 5 == 0]

    result = sort_dic(json.load(open('/work/mayixiao/cutoff/preresults/LeCaRD/%s.json'%(args.dataset_name), 'r')))

elif args.data == 2:
    with open('/work/mayixiao/similar_case/202006/data/label/label_top30_dict_2.json', 'r') as f:
        label_dic = json.load(f)
    keys_train = list(json.load(open('/work/mayixiao/similar_case/202006/data/label/label_final.json', 'r')).keys())
    keys = list(json.load(open('/work/mayixiao/similar_case/202006/data/label/label_big.json', 'r')).keys())

    result = json.load(open('/work/mayixiao/cutoff/preresults/CAIL2021/%s.json'%(args.dataset_name), 'r'))

elif args.data == 3:
    label_dic = json.load(open('/work/mayixiao/cutoff/preresults/COLIEE2020/labels.json', 'r'))
    keys_train = [str(i).rjust(3,'0') for i in range(1,521) if i%5 != 0]
    keys = [str(i) for i in range(521, 651)]
    
    result = json.load(open('/work/mayixiao/cutoff/preresults/COLIEE2020/%s.json'%(args.dataset_name), 'r'))

# dics = [cut_K(tfidf,20), bm25, lmir, base, rbert, lfm_base, cut_K(l2r,25)]
# print(bm25)

if args.data <= 2:
    dics = [cut_K(result,100), cut_K(result,5),cut_K(result,10),cut_K(result,30),cut_K(result,50)]
else:
    dics = [cut_K(result,30), cut_K(result,5), cut_K(result,10), cut_K(result,15), cut_K(result,20)]

# round4(eval_metrics(args, keys, label_dic, dics))
# greedy(args, keys, keys_train,label_dic, dics)
round4([[oracle(args, result, keys, label_dic)]])

# dics = [cut_K(result,30), cut_K(result,5), cut_K(result,10), cut_K(result,15), cut_K(result,20), cut_K(result, 6)]
# dics = [cut_K(result,100), cut_K(result,5),cut_K(result,10),cut_K(result,30),cut_K(result,50), cut_K(result, 24)]

# scores_list_raw = eval_metrics(args, keys, label_dic, dics)
# scores_list = [sublist[0] for sublist in scores_list_raw]
# scores_list.append()
# # print(scores_list)
# ttests = []
# for i in range(len(scores_list[:-1])):
#     p = stats.wilcoxon(scores_list[-1], scores_list[i], correction=False, alternative='greater', mode='approx').pvalue
#     # ttests.append(p)
#     if p < 0.01:
#         ttests.append('dd')
#     elif p < 0.05:
#         ttests.append('d')
#     else:
#         ttests.append(0)
        
# print( ttests)








