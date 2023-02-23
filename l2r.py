import argparse
from pyexpat import features
import lightgbm as lgb
import os
import json
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, ndcg_score
import math


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
        return torch.ones(args.dim).tolist()
        # return -1

def init(args):
    dic = {}
    lines = open('/work/mayixiao/www22/node2vec/emb/crime_2.emb', 'r').readlines()[1:]
    for line in lines:
        tem_list = line.strip().split(' ')
        key = tem_list[0]
        value = [float(i) for i in tem_list[1:]]
        dic[key] = value

    id2cos = {}
    if args.data == 1:
        path1 = '/work/mayixiao/cutoff/train_short_top100.json'
        path2 = '/work/mayixiao/cutoff/test_short_top100.json'
    elif args.data == 2:
        path1 = '/work/mayixiao/cutoff/train_2_short_top100.json'
        path2 = '/work/mayixiao/cutoff/test_2_short_top100.json'
    lines1 = open(path1, 'r').readlines()[:]
    lines2 = open(path2, 'r').readlines()[:]
    lines = lines1 + lines2
    for line in lines:
        q_embd = get_embd(eval(line)['c_a'], dic)
        c_embd = get_embd(eval(line)['c_b'], dic)
        if q_embd == -1 or c_embd == -1:
            id2cos[eval(line)['guid']] = 0.3
        else:
            id2cos[eval(line)['guid']] = cos(q_embd, c_embd)

    return id2cos



def init_dataset(args, lines, label_dic, cutoff_scores):
    # scores = [score[0] for cutoff_score in cutoff_scores for score in cutoff_score ]
    # assert len(lines) == len(scores)
    x_list, y_list = [], []
    top100_dic = {}
    group = []
    count = 0
    pre_qid = -1
    rel_num_dic = {}
    for key1 in cutoff_scores:
        rel_num_dic[key1] = [1]
        key2s = list(cutoff_scores[key1].keys())
        for key2 in cutoff_scores[key1]:
            index = key2s.index(key2)
            if index >= 1:
                if cutoff_scores[key1][key2] - cutoff_scores[key1][key2s[index-1]] > 0:
                    rel_num_dic[key1].append(rel_num_dic[key1][-1]+1)
                else:
                    rel_num_dic[key1].append(rel_num_dic[key1][-1])
                # if key2 in label_dic[key1] and label_dic[key1][key2] >= 1:
                #     rel_num_dic[key1].append(rel_num_dic[key1][-1]+1)
                # else:
                #     rel_num_dic[key1].append(rel_num_dic[key1][-1])

    # for i, line in enumerate(lines):
    #     dic = eval(line)
    #     qid = dic['id_'].split('_')[0]
    #     cid = dic['id_'].split('_')[1]
    #     if 1:
    for qid in lines:
        for cid in lines[qid]:
            if qid not in top100_dic:
                top100_dic[qid] = [cid]
            else:
                top100_dic[qid].append(cid)

            if pre_qid == qid:
                count += 1
            else:
                if pre_qid != -1:
                    group.append(count)
                pre_qid = qid
                count = 1
            
            features = [lines[qid][cid]]
            # features = dic['res']
            # features = [dic['res'][1] - dic['res'][0]]
            if args.data <= 2:
                # features.append(id2cos[dic['id_']])
                features.append(id2cos[qid+'_'+cid])
            if args.iter >=2:
                keys = list(cutoff_scores[qid].keys())
                index = keys.index(cid)
                
                if index == 0:
                    features.append(cutoff_scores[qid][cid])
                    # features.extend([cutoff_scores[qid][cid], 0])
                else:
                    N = rel_num_dic[qid][-1]
                    # a = cutoff_scores[qid][cid] 
                    a = cutoff_scores[qid][cid] - cutoff_scores[qid][keys[index-1]] #1e-9的误差会对max_depth=none的模型造成结果差异！
                    # print(a)
                    # a = rel_num_dic[qid][index-1]*( math.log(cutoff_scores[qid][cid]/cutoff_scores[qid][keys[index-1]])  *(N+index)/(N+index-1) -1)
                    # a = rel_num_dic[qid][index-1] / (N+index-1) + 0.5*math.log(cutoff_scores[qid][cid]/cutoff_scores[qid][keys[index-1]])*(N+index)
                    # a = math.log(cutoff_scores[qid][cid]/cutoff_scores[qid][keys[index-1]])
                    # a = cutoff_scores[qid][cid]/cutoff_scores[qid][keys[index-1]]
                    # a = rel_num_dic[qid][index-1]*( math.log(cutoff_scores[qid][cid]/cutoff_scores[qid][keys[index-1]]) -1 )
                    # print(qid, cid, rel_num_dic[qid][index]-rel_num_dic[qid][index-1], a)
                    features.append(a)
                    # features.extend([cutoff_scores[qid][cid], cutoff_scores[qid][keys[index-1]]])
                # features.append(0)
            
            if args.data <= 2:
                if cid in label_dic[qid]:
                    label = label_dic[qid][cid]//2 #????
                else:
                    label = 0
            else:
                label = label_dic[qid][cid]
            # print(qid, cid, index, label)
            x_list.append(features)
            y_list.append(label)
    group.append(count)

    # print(len(x_list), sum(group))
    # print(group)
    return x_list, y_list, group, top100_dic

def write_scores(top100_dic, group, y_pred, wpath):
    w_dict = {}
    keys = list(top100_dic.keys())
    now = 0
    for idx, new in enumerate(group):
        qid = keys[idx]
        w_dict[qid] = {}
        for i in range(new):
            w_dict[qid][top100_dic[qid][i]] = y_pred[now+i]
        now += new

    json.dump(w_dict, open(wpath, 'w'), ensure_ascii=False)
    return
    

def learning2rank(args, X_train, X_test, y_train, y_test, group_train, group_test, top100_dic_train, top100_dic_test):
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    # validation_data = lgb.Dataset(X_test, label=y_test, group=group_test)
    # validation_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    
    if args.data <=2:
        target = [10,20,30,100]
    else:
        target = [1,3,5,10]
    params = {
        # 'learning_rate': 0.001,
        # 'lambda_l1': 0.1,
        # 'lambda_l2': 0.2,
        'max_depth': 1 if args.data==3 else None,
        'num_iterations': 100 if args.data==3 else 30,
        'ndcg_at': target,#[10,20,30,100],
        'objective': 'lambdarank',  #
        # 'num_class': 3,
        'metric': 'NDCG'
    }

    gbm = lgb.train(params, train_data)#, valid_sets=[validation_data])
    y_pred_test = gbm.predict(X_test)
    y_pred_train = gbm.predict(X_train)
    
    for k in target:
        now = 0
        ndcg_all = []
        for new in group_test:
            
            ndcg_all.append(ndcg_score([y_test[now:now + new]], [y_pred_test[now:now + new]], k=k))
            now += new
            
        
        print(sum(ndcg_all)/len(ndcg_all))
        # if k == 30:
            # print([(key, ndcg) for key, ndcg in zip(list(top100_dic_test.keys()), ndcg_all)] )


    if args.data == 1:
        wpath_train = '/work/mayixiao/cutoff/results/l2r_LeCaRD_%s_train_%i.json'%(args.dataset_name, args.iter)
        wpath_test = '/work/mayixiao/cutoff/results/l2r_LeCaRD_%s_test_%i.json'%(args.dataset_name, args.iter)
    elif args.data == 2:
        wpath_train = '/work/mayixiao/cutoff/results/l2r_CAIL2021_%s_train_%i.json'%(args.dataset_name, args.iter)
        wpath_test = '/work/mayixiao/cutoff/results/l2r_CAIL2021_%s_test_%i.json'%(args.dataset_name, args.iter)
    elif args.data == 3:
        wpath_train = '/work/mayixiao/cutoff/results/l2r_COLIEE2020_%s_train_%i.json'%(args.dataset_name, args.iter)
        wpath_test = '/work/mayixiao/cutoff/results/l2r_COLIEE2020_%s_test_%i.json'%(args.dataset_name, args.iter)

    write_scores(top100_dic_train, group_train, y_pred_train, wpath_train)
    write_scores(top100_dic_test, group_test, y_pred_test, wpath_test)

    return sum(ndcg_all)/len(ndcg_all)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=";2R Args")
    parser.add_argument('--data', type=int, default=1)
    parser.add_argument('--model', type=str, default='lecut')
    parser.add_argument('--dataset-name', type=str, default='slr')
    parser.add_argument('--iter', type=int, required=True)
    parser.add_argument('--dim', type=int, default=16)


    args = parser.parse_args()

    if args.data <= 2:
        id2cos = init(args)
        print('init id2cos done.')

    if args.data == 1:
        label_dic = json.load(open('/work/mayixiao/similar_case/LeCaRD/LeCaRD_github/data/label/label_top30_dict.json', 'r'))
        if args.iter >= 2:
            cutoff_scores= json.load(open('/work/mayixiao/cutoff/results/%s_LeCaRD_%s_%i.json'%(args.model, args.dataset_name,args.iter-1), 'r'))
            train_lines = json.load(open('/work/mayixiao/cutoff/results/l2r_LeCaRD_%s_train_%i.json'%(args.dataset_name, args.iter-1), 'r'))
            test_lines = json.load(open('/work/mayixiao/cutoff/results/l2r_LeCaRD_%s_test_%i.json'%(args.dataset_name, args.iter-1), 'r'))
            # train_lines = open('/work/mayixiao/cutoff/preresults/LeCaRD/scores/LeCaRD_%s_train_top100.json'%(args.dataset_name), 'r').readlines()
            # test_lines = open('/work/mayixiao/cutoff/preresults/LeCaRD/scores/LeCaRD_%s_test_top100.json'%(args.dataset_name), 'r').readlines()
        else:
            train_lines = open('/work/mayixiao/cutoff/preresults/LeCaRD/scores/LeCaRD_%s_train_top100.json'%(args.dataset_name), 'r').readlines()
            test_lines = open('/work/mayixiao/cutoff/preresults/LeCaRD/scores/LeCaRD_%s_test_top100.json'%(args.dataset_name), 'r').readlines()
            cutoff_scores = {}

    elif args.data == 2:
        train_lines = open('/work/mayixiao/cutoff/preresults/CAIL2021/scores/CAIL2021_%s_train_top100.json'%(args.dataset_name), 'r').readlines()
        test_lines = open('/work/mayixiao/cutoff/preresults/CAIL2021/scores/CAIL2021_%s_test_top100.json'%(args.dataset_name), 'r').readlines()
        label_dic = json.load(open('/work/mayixiao/similar_case/202006/data/label/label_top30_dict_2.json', 'r'))
        if args.iter >= 2:
            cutoff_scores= json.load(open('/work/mayixiao/cutoff/results/%s_CAIL2021_%s_%i.json'%(args.model, args.dataset_name, args.iter-1), 'r'))
            train_lines = json.load(open('/work/mayixiao/cutoff/results/l2r_CAIL2021_%s_train_%i.json'%(args.dataset_name, args.iter-1), 'r'))
            test_lines = json.load(open('/work/mayixiao/cutoff/results/l2r_CAIL2021_%s_test_%i.json'%(args.dataset_name, args.iter-1), 'r'))
        else:
            train_lines = open('/work/mayixiao/cutoff/preresults/CAIL2021/scores/CAIL2021_%s_train_top100.json'%(args.dataset_name), 'r').readlines()
            test_lines = open('/work/mayixiao/cutoff/preresults/CAIL2021/scores/CAIL2021_%s_test_top100.json'%(args.dataset_name), 'r').readlines()
            cutoff_scores = {}
    elif args.data == 3:
        label_dic = json.load(open('/work/mayixiao/cutoff/preresults/COLIEE2020/labels.json', 'r'))
        if args.iter >= 2:
            cutoff_scores= json.load(open('/work/mayixiao/cutoff/results/%s_COLIEE2020_%s_%i.json'%(args.model, args.dataset_name, args.iter-1), 'r'))
            train_lines = json.load(open('/work/mayixiao/cutoff/results/l2r_COLIEE2020_%s_train_%i.json'%(args.dataset_name, args.iter-1), 'r'))
            test_lines = json.load(open('/work/mayixiao/cutoff/results/l2r_COLIEE2020_%s_test_%i.json'%(args.dataset_name, args.iter-1), 'r'))
        else:
            train_lines = open('/work/mayixiao/cutoff/preresults/COLIEE2020/scores/COLIEE2020_%s_train_top30.json'%(args.dataset_name), 'r').readlines()
            test_lines = open('/work/mayixiao/cutoff/preresults/COLIEE2020/scores/COLIEE2020_%s_test_top30.json'%(args.dataset_name), 'r').readlines()
            cutoff_scores = {}
    X_train, y_train, group_train, top100_dic_train = init_dataset(args, train_lines, label_dic, cutoff_scores)
    X_test, y_test, group_test, top100_dic_test = init_dataset(args, test_lines, label_dic, cutoff_scores)
    learning2rank(args, X_train, X_test, y_train, y_test, group_train, group_test, top100_dic_train, top100_dic_test)