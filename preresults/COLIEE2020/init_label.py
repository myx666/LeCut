import os
import json
from tqdm import tqdm

# lines_train = open('/work/mayixiao/coliee_2020/data/task1/format/body/train_top30.json').readlines()
lines_test = open('/work/mayixiao/coliee_2020/data/task1/format/body/test_top30.json').readlines()

# w_dic = {}
# for line in tqdm(lines_train):
#     dic = eval(line)
#     qid, cid = dic['guid'].split('_')
#     label = dic['label']
#     if qid in w_dic:
#         w_dic[qid][cid] = label
#     else:
#         w_dic[qid] = {cid:label}

# test_labels = json.load(open('/work/mayixiao/coliee_2020/data/task1/task1_test_2020_labels.json', 'r'))

# for line in tqdm(lines_test):
#     dic = eval(line)
#     qid, cid = dic['guid'].split('_')
#     label = int(cid+'.txt' in test_labels[qid])
#     if qid in w_dic:
#         w_dic[qid][cid] = label
#     else:
#         w_dic[qid] = {cid:label}

# json.dump(w_dic, open('/work/mayixiao/cutoff/coliee2020/data/labels.json', 'w'), ensure_ascii=False)

labels = json.load(open('/work/mayixiao/cutoff/coliee2020/data/labels.json', 'r'))
w_lines = []
for line in lines_test:
    dic = eval(line)
    qid, cid = dic['guid'].split('_')
    dic['label'] = labels[qid][cid]
    w_lines.append(dic)

with open('/work/mayixiao/coliee_2020/data/task1/format/body/test_top30.json', 'w') as f:
    for line in w_lines:
        json.dump(line, f, ensure_ascii=False)
        f.write('\n') 