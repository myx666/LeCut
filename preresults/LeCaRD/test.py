import json
from tqdm import tqdm
file_ = json.load(open('/work/mayixiao/cutoff/preresults/LeCaRD/embd_slr_lecard.json','r'))
# file_ = open('/work/mayixiao/cutoff/preresults/LeCaRD/embd_slr_lecard.json','r').read()
# w_lines = []
# for line in tqdm(file_[:3]):
    # w_lines.append(line)

# json.dump(w_lines, open('/work/mayixiao/cutoff/preresults/LeCaRD/embd_slr_lecard2.json','w'), ensure_ascii=False, indent=2)
with open('/work/mayixiao/cutoff/preresults/LeCaRD/embd_slr_lecard2.json','w') as f:
    for line in tqdm(file_):
        json.dump(line, f, ensure_ascii=False)
        f.write('\n')