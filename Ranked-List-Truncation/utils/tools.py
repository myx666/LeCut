# -*- encoding: utf-8 -*-
'''
@Func    :   tool 
@Time    :   2021/12/15 16:33:15
@Author  :   Yixiao Ma 
@Contact :   mayx20@mails.tsinghua.edu.cn
'''

import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

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
    lines = json.load(open(path, 'r'))
    # w_dic = defaultdict(lambda: defaultdict (lambda: 0.0))
    w_dic = {}
    for dic in lines:
        qid, cid = dic['id_'].split('_')
        if qid not in w_dic:
            w_dic[qid] = {cid:dic['embd']}
        else:
            w_dic[qid][cid] = dic['embd']
    return w_dic

if __name__ == '__main__':
    w_dic = process_embd('/work/mayixiao/cutoff/preresults/CAIL2021/embd_slr_cail2021.json')
    # print(w_dic['4891']['24048'])
    # json.dump(w_dic, open('/work/mayixiao/cutoff/preresults/LeCaRD/embd_slr_lecard_dic.json'))
    