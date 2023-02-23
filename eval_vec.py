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

def get_embd(c_list, dic):
    tensors = torch.FloatTensor([dic[i] for i in c_list if i in dic])
    if tensors.size()[0] != 0:
        return tensors.mean(0).tolist()
    else:
        return torch.randn(32).tolist()
        # return [0.0001] * 32

def init(data):
    dic = {}
    lines = open('/work/mayixiao/www22/node2vec/emb/crime.emb', 'r').readlines()[1:]
    for line in lines:
        tem_list = line.strip().split(' ')
        key = tem_list[0]
        value = [float(i) for i in tem_list[1:]]
        dic[key] = value

    id2cos = {}
    if data == 1:
        path = '/work/mayixiao/www22/test.json'
    elif data == 2:
        path = '/work/mayixiao/www22/test_2_short.json'
    lines = open(path, 'r').readlines()[:]
    for line in lines:
        q_embd = get_embd(eval(line)['c_a'], dic)
        c_embd = get_embd(eval(line)['c_b'], dic)
        id2cos[eval(line)['guid']] = cos(q_embd, c_embd)

    return id2cos

def dcg(ranks, pos_scale):
    dcg_value = 0.

    for i,l in enumerate(ranks):
        logi = math.log(i+2,2)
        if l >=pos_scale:
            dcg_value += 1 / logi
        else:
            dcg_value += -1 / logi
        
    return dcg_value

def nci(ranks, pos_scale, data):
    alpha = {1: 221.1, 2:171.6, 3:49.0}[data]
    nci_value = 0.

    for i,l in enumerate(ranks):
        logi = math.log(i+2,2)
        if l >=pos_scale:
            nci_value += 1 / logi
        else:
            nci_value += -(i+1) / alpha
        
    return nci_value

def ndcg(ranks, gt_ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    # log_ki = []

    sranks = sorted(gt_ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi

    return dcg_value/idcg_value

def f1(ranks, rels, labels, pos_scale):
    if len(rels) != 0:
        precision = len([i for i in ranks if (str(i) in labels and labels[str(i)] >= pos_scale)])/len(ranks)
        recall = len([i for i in ranks if (str(i) in labels and labels[str(i)] >= pos_scale)])/len(rels)
        if precision + recall == 0:
            score = 0.0
        else:
            score = 2*precision*recall / (precision + recall)
    else:
        score = 0.0
    return score

def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 0, keepdims = True)
    s = x_exp / x_sum    
    return s

def get_rankdict(vec_lines):#, K, id2cos): 暂时不需要id2cos
    w_dict = {}
    tem_dict = {}           
    for line in vec_lines:
        guid = eval(line)['id_']
        q = guid.split('_')[0]
        if q not in tem_dict:
            tem_dict[q] = {}
        idx = int(guid.split('_')[1])
        c = str(idx)
        # if args.mode == 'cls':

        tem_dict[q][c] = eval(line)['res'][1] - eval(line)['res'][0] #+ K * id2cos[eval(line)['id_']]
            # if q == '84556':
                # print(c, eval(line)['res'][1] - eval(line)['res'][0], eval(line)['res'][1] - eval(line)['res'][0] + K * id2cos[eval(line)['id_']])
                
        # elif args.mode == '4cls':
        #     probs = eval(line)['res']
        #     maxidx = probs.index(max(probs))
        #     s_probs = sorted(probs, reverse=True)
        #     tem_dict[q][c] = maxidx*10000 + eval(line)['res'][3] + eval(line)['res'][2] - eval(line)['res'][1] - eval(line)['res'][0]
        # elif args.mode == 'reg':
        #     tem_dict[q][c] = eval(line)['res'][0]
        
    for q in tem_dict:
        rank_items = sorted(tem_dict[q].items(), key = lambda x: x[1], reverse=True )
        w_dict[q] = [i[0] for i in rank_items]

    return w_dict
    # json.dump(w_dict, open(WPATH, 'w'), ensure_ascii=False)

def eval_metrics(args, keys, label_dic, dics):
    score_list = []
    all_score_list = []
    all_ranks = []
    for dic in dics:
        tem_score_list = []
        tem_all_score_list = []
        tem_rank = []
        if args.m == 'NDCG':
            topK_list = [10, 20, 30, 100]
            # topK_list = [30]
        elif args.m == 'P':
            topK_list = [5,10]
        elif args.m == 'MAP' or args.m == 'F1' or args.m == 'DCG' or args.m == 'NCI':
            topK_list = [1]
        for topK in topK_list:
            temK_all_score_list = []
            s_score = 0.0
            for key in keys[:]:
                if args.m == 'NDCG':
                    rawranks = []
                    # print(key)
                    for i in dic[key]:
                        if str(i) in label_dic[key]:
                            rawranks.append(label_dic[key][str(i)])
                        else:
                            rawranks.append(0)
                    
                    ranks = rawranks
                    # ranks = rawranks + [0]*(100-len(rawranks))
                    if sum(ranks) != 0:
                        score = ndcg(ranks, list(label_dic[key].values()) + [0]*(100-len(list(label_dic[key].values()))) , topK)
                    else:
                        score = 0.0

                    tem_rank.append(ranks)
                    
                elif args.m == 'P':
                    ranks = [i for i in dic[key] if str(i) in list(label_dic[key].keys())]
                    score = float(len([j for j in ranks[:topK] if label_dic[key][str(j)] == 3])/topK)
                    
                elif args.m == 'MAP':
                    ranks = [i for i in dic[key] if str(i) in list(label_dic[key].keys())] 
                    rels = [ranks.index(i) for i in ranks if label_dic[key][str(i)] == 3]
                    tem_map = 0.0
                    for rel_rank in rels:
                        tem_map += float(len([j for j in ranks[:rel_rank+1] if label_dic[key][str(j)] == 3])/(rel_rank+1))
                    if len(rels) > 0:
                        score = tem_map / len(rels)
                    else:
                        score = 0.0

                elif args.m == 'F1':
                    # ranks = [i for i in dic[key] if str(i) in list(label_dic[key].keys())] 
                    ranks = dic[key]
                    pos_scale = 2 if args.data <= 2 else 1
                    rels = [i for i in label_dic[key] if label_dic[key][i] >= pos_scale]
                    score = f1(ranks, rels, label_dic[key], pos_scale)
                
                elif args.m == 'DCG':
                    # ranks = [label_dic[key][str(i)] for i in dic[key] if str(i) in label_dic[key]]
                    ranks = []
                    for i in dic[key]:
                        if str(i) in label_dic[key]:
                            ranks.append(label_dic[key][str(i)])
                        else:
                            ranks.append(0)
                    if sum(ranks) != 0:
                        pos_scale = 2 if args.data <= 2 else 1
                        score = dcg(ranks, pos_scale)
                        
                    else:
                        score = 0.0

                elif args.m == 'NCI':
                    
                    ranks = []
                    for i in dic[key]:
                        if str(i) in label_dic[key]:
                            ranks.append(label_dic[key][str(i)])
                        else:
                            ranks.append(0)
                    if sum(ranks) != 0:
                        pos_scale = 2 if args.data <= 2 else 1
                        score = nci(ranks, pos_scale, args.data)
                        
                    else:
                        score = 0.0

                s_score += score
                temK_all_score_list.append(score)
            tem_score_list.append(s_score/len(keys))
            tem_all_score_list.append(temK_all_score_list)
        score_list.append(tem_score_list)
        all_score_list.append(tem_all_score_list)
        all_ranks.append(tem_rank)
    # print('%s: '%(args.m), score_list)
    # print(all_score_list) 
    return score_list 
    # return all_score_list

    # for base_scores in all_score_list[:-1]:
    #     ttests = []
    #     for i in range(len(topK_list)):
    #         # print(stats.levene(all_score_list[i], base_scores[i]))
    #         # ttests.append(stats.ttest_ind(all_score_list[i], base_scores[i], equal_var=False).pvalue)  
    #         ttests.append(stats.wilcoxon(all_score_list[-1][i], base_scores[i], correction=True, alternative='greater', mode='approx').pvalue)  
    #     print('pvalue: ', ttests)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Help info:")
    parser.add_argument('--vec', type=str, required=True, help='vector file path.')
    parser.add_argument('--mode', type=str, default='cls', help='classification or regression')
    parser.add_argument('--data', type=int, choices=[1,2], default=1, help='which dataset')
    parser.add_argument('--K', type=float, default=1, help='hyper para')
    parser.add_argument('--m', type=str, choices= ['NDCG', 'P', 'MAP', 'KAPPA'], default='NDCG', help='Metric.')

    args = parser.parse_args()

    id2cos = init()

    if args.data == 1:
        with open('/data/home/scv2421/run/lecard/data/label/label_top30_dict.json', 'r') as f:
            label_dic = json.load(f)
        keys = [i for i in list(label_dic.keys())[:100] if list(label_dic.keys())[:100].index(i) % 5 == 0]
        # base_ndcgs_list = [lecard_base, lfm_lecard_base]
    elif args.data == 2:
        with open('/data/home/scv2421/run/www22/label_top30_dict_2.json', 'r') as f:
            label_dic = json.load(f)
        keys = list(json.load(open('/data/home/scv2421/run/www22/label_big.json', 'r')).keys())
        # base_ndcgs_list = [cail_base, lfm_cail_base]
        
    ROOT = '/data/home/scv2421/run/pytorch_worker/result/'
    VECPATH = os.path.join(ROOT, args.vec)
    WPATH = '/work/mayixiao/similar_case/LeCaRD/private/data/bert.json'
    vec_lines = open(VECPATH, 'r').readlines()

    if args.data == 1:
        bm25 = json.load(open('/data/home/scv2421/run/www22/prediction_bm25.json', 'r'))
        lmir = json.load(open('/data/home/scv2421/run/www22/prediction_lmir.json', 'r'))
        tfidf = json.load(open('/data/home/scv2421/run/www22/prediction_tfidf.json', 'r'))
        rbert = open(os.path.join(ROOT, 'rbert_LeCaRD_base_4.json'), 'r').readlines()
        base = open(os.path.join(ROOT, 'LeCaRD_base_3.json'), 'r').readlines()
        lfm_base = open(os.path.join(ROOT, 'lfm_LeCaRD_base_4.json'), 'r').readlines()
        l2r = json.load(open(os.path.join(ROOT, 'l2r_best.json'), 'r'))

    elif args.data == 2:
        bm25 = json.load(open('/data/home/scv2421/run/www22/bm25_2.json', 'r'))
        lmir = json.load(open('/data/home/scv2421/run/www22/lmir_2.json', 'r'))
        tfidf = json.load(open('/data/home/scv2421/run/www22/tfidf_2.json', 'r'))
        rbert = open(os.path.join(ROOT, 'rbert_CAIL_base_2.json'), 'r').readlines()
        base = open(os.path.join(ROOT, 'CAIL_base_3.json'), 'r').readlines()
        lfm_base = open(os.path.join(ROOT, 'lfm_CAIL_base_5.json'), 'r').readlines()
        l2r = json.load(open(os.path.join(ROOT, 'l2r_2_best.json'), 'r'))

    # dics = [tfidf, bm25, lmir, get_rankdict(base, 0), get_rankdict(lfm_base, 0), get_rankdict(vec_lines, args.K)]
    dics = [get_rankdict(vec_lines, args.K)]#, get_rankdict(vec_lines, 0) ,get_rankdict(vec_lines, args.K)]
    # dics = [tfidf, bm25, lmir, get_rankdict(base, 0), get_rankdict(rbert, 0) ,get_rankdict(lfm_base, 0), l2r]
    eval_metrics(args.m, keys, label_dic, dics)