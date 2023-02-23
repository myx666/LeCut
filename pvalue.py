import os
import json
from scipy import stats
import numpy as np
from tqdm import tqdm


root = '/work/mayixiao/cutoff/results/sig'
files = os.listdir(root)

def sort_values(data_raw):
    sorted_results = sorted(data_raw.items(), key=lambda x: x[0])
    return [item[1] for item in sorted_results]

def fisher_rand(alist, blist):
    assert(len(alist)==len(blist))
    all = np.stack([alist,blist])
    all = all.reshape(-1,2)
    leng = len(alist)
    randnum = 10000
    posnum = 0
    sumab = sum(alist) + sum(blist)
    delta = sum(alist) - sum(blist)
    for i in range(randnum):
        rdn = np.random.randint(0,2,leng)
        newa = [d[j] for d,j in zip(all,rdn)]
        # print(sum(newa), sumab, delta)
        if 2*sum(newa) - sumab > delta:
            posnum += 1
    # print(posnum/randnum)
    return posnum/randnum


def get_pvalue(files, dataset, model, m):
    all_score_list = []
    
    cutoffs = ['bicut', 'choppy', 'attncut', 'lecut']
    # cutoffs = ['lecut']
    for cutoff in cutoffs:
        name = cutoff + '_' + dataset + '_' + model
        tem_files = [file_ for file_ in files if name in file_]
        assert len(tem_files) == 1
        all_score_list.append(sort_values(json.load(open(os.path.join(root, tem_files[0])))[m]))
    
    # print(all_score_list[0])
    
    ttests = []
    for i in range(len(all_score_list[:-1])):
        # print(sum(all_score_list[i]))
        # print(stats.levene(all_score_list[-1], all_score_list[i]))
        # print(stats.shapiro(all_score_list[i]))
        # ttests.append(stats.ttest_ind(all_score_list[-1], all_score_list[i], equal_var=True).pvalue)  
        # ttests.append(stats.mannwhitneyu(all_score_list[-1], all_score_list[i], alternative='greater').pvalue)
        # p = stats.wilcoxon(all_score_list[-1], all_score_list[i], correction=False, alternative='greater', mode='approx').pvalue
        p = fisher_rand(all_score_list[-1], all_score_list[i])
        
        ttests.append(p)
        # if p < 0.01:
        #     ttests.append('dd')
        # elif p < 0.05:
        #     ttests.append('d')
        # else:
        #     ttests.append(0)
        
    print(dataset, model, m, ttests)
    return


datasets = ['LeCaRD', 'CAIL2021', 'COLIEE2020']
models = ['slr', 'bert', 'roberta']
ms = ['f1', 'dcg', 'nci']

for dataset in datasets[2:]:
    for model in models[1:]:
        for m in ms:
            get_pvalue(files, dataset, model, m)

# if __name__ == '__main__':
    # fisher_rand([1,0,3,0],[2,2,2,1])