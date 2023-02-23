# -*- encoding: utf-8 -*-
'''
@Func    :   get statistics of different datasets
@Time    :   2021/12/08 21:37:48
@Author  :   Yixiao Ma 
@Contact :   mayx20@mails.tsinghua.edu.cn
'''

import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import jieba

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

def add_doc_len(data, stats_={}):
    # # lecard
    stats = stats_
    if data == 'LeCaRD':
        gt = json.load(open('/work/mayixiao/similar_case/LeCaRD/LeCaRD_github/data/label/label_top30_dict.json','r'))
        dir_path = '/work/mayixiao/similar_case/candidates'
    elif data == 'CAIL2021':
        gt = json.load(open('/work/mayixiao/similar_case/202006/data/label/label_top30_dict_2.json','r'))
        dir_path = '/work/mayixiao/similar_case/202006/data/candidates_2'
    elif data == 'COLIEE2020':
        gt = json.load(open('/work/mayixiao/cutoff/preresults/COLIEE2020/labels.json','r'))
        dir_path = '/work/mayixiao/coliee_2020/data/task1/task1_2020'
        
    keys = list(gt.keys())
    
    for key1 in tqdm(keys):
        stats[key1] = {}
        if data == 'COLIEE2020':
            files = os.listdir(os.path.join(dir_path, key1, 'candidates'))
        else:
            files = os.listdir(os.path.join(dir_path, key1))
        key2s = list(gt[key1].keys())
        if data == 'COLIEE2020':
            for key2 in key2s:
                doc = open(os.path.join(dir_path, key1, 'candidates', key2 + '.txt'),'r').read()
                stats[key1][key2] = {'doclen': len(doc.split(' '))}
        else:
            for key2 in key2s: #list(gt[key1].keys())[:30]:
                stats[key1][key2] = {'doclen': len(json.load(open(os.path.join(dir_path, key1, key2 + '.json'),'r'))['qw'])}

    return stats

def split_word(line, stopwords):
    raw_a = jieba.cut(line, cut_all=False)
    a = [i for i in raw_a if not i in stopwords]
    return  " ".join(a)
    
def add_neighbor_tfidf(data, stats_={}):
    stats = stats_
    vectorizer = CountVectorizer()  # 实例化
    transformer = TfidfTransformer()
    if data == 'LeCaRD':
        gt = json.load(open('/work/mayixiao/similar_case/LeCaRD/LeCaRD_github/data/label/label_top30_dict.json','r'))
        dir_path = '/work/mayixiao/similar_case/candidates'
    elif data == 'CAIL2021':
        gt = json.load(open('/work/mayixiao/similar_case/202006/data/label/label_top30_dict_2.json','r'))
        dir_path = '/work/mayixiao/similar_case/202006/data/candidates_2'
    elif data == 'COLIEE2020':
        gt = json.load(open('/work/mayixiao/cutoff/preresults/COLIEE2020/labels.json','r'))
        dir_path = '/work/mayixiao/coliee_2020/data/task1/task1_2020'

    keys = list(gt.keys())

    for key1 in tqdm(keys):
        corpus = []

        if data == 'COLIEE2020':
            files = os.listdir(os.path.join(dir_path, key1, 'candidates'))
            key2s = list(gt[key1].keys())
            for key2 in key2s:
                doc = open(os.path.join(dir_path, key1, 'candidates', key2 + '.txt'),'r').read()[:4096]
                corpus.append(doc)
        else:
            files = os.listdir(os.path.join(dir_path, key1))
            key2s = [file_.split('.')[0] for file_ in files]
            for key2 in key2s:
                corpus.append(split_word(json.load(open(os.path.join(dir_path, key1, key2 + '.json'),'r'))['ajjbqk'], stopwords))
        tfidf_list = transformer.fit_transform(vectorizer.fit_transform(corpus)).toarray().tolist()

        num_file = len(key2s)
        for i in range(num_file):
            stats[key1][key2s[i]]['neighbor_tfidf'] = {}
            for j in range(num_file):
                stats[key1][key2s[i]]['neighbor_tfidf'][key2s[j]] = cos(tfidf_list[i], tfidf_list[j])

    return stats

if __name__ == '__main__':
    with open('/work/mayixiao/similar_case/LeCaRD/LeCaRD_github/data/others/stopword.txt', 'r') as g:
        words = g.readlines()
    stopwords = [i.strip() for i in words]
    stopwords.extend(['.','（','）','-'])

    DATA = 'COLIEE2020'
    dic = {}
    dic = add_doc_len(DATA, dic)
    dic = add_neighbor_tfidf(DATA, dic)
    WPATH = '/work/mayixiao/cutoff/preresults/%s/stats.json'%(DATA)
    json.dump(dic, open(WPATH, 'w'), ensure_ascii=False)
