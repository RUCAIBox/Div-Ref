import sys
import json
import numpy as np


def compute_score(results, metric, ref_idx, ref_num, detail=False):
    hyp0_scores = []
    hyp1_scores = []
    types = []
    labels = []
    for result in results:
        types.append(result['type'])
        labels.append(result['label'])
        hyp0_score = result['hyp0']['metric_score']
        hyp1_score = result['hyp1']['metric_score']
        if ref_idx is not None:
            hyp0_score = np.array(hyp0_score)[ref_idx]
            hyp1_score = np.array(hyp1_score)[ref_idx]
        hyp0_score = {k: aggr_func([s[k] for s in hyp0_score[:ref_num+1]]) for k in hyp0_score[0]}
        hyp1_score = {k: aggr_func([s[k] for s in hyp1_score[:ref_num+1]]) for k in hyp1_score[0]}
        hyp0_scores.append(hyp0_score)
        hyp1_scores.append(hyp1_score)

    tmp_hyp0_scores = np.array([s[metric] for s in hyp0_scores])
    tmp_hyp1_scores = np.array([s[metric] for s in hyp1_scores])
    labels = np.array(labels)
    types = np.array(types)
    acc_state = ((tmp_hyp0_scores >= tmp_hyp1_scores) & (labels == 1)) | ((tmp_hyp0_scores < tmp_hyp1_scores) & (labels == -1))
    accs = []
    for i in range(1, 5):
        accs.append(float(acc_state[types == i].sum() / 50))

    if detail:
        for acc in accs:
            print(round(acc, 2), end = ' ')
        print()
    return np.mean(accs)

filename = sys.argv[1]
results = json.load(open(filename))
aggr_func = np.mean if len(sys.argv) == 2 or sys.argv[2] == 'mean' else np.max

direction = filename.split('.')[1]

metric_group = []
human_group = []

ref_num = len(results[0]['refs'])
sample_num = len(results)

random_idx = np.array([0,1,11,21,31,41,51,61,71,81,91])

for metric in ['bleu', 'rouge', 'meteor', 'cider', 'bertscore', 'spice']:
    print(metric)
    print(compute_score(results, metric, random_idx, 10, detail=True))