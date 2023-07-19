import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
from scipy import stats
from tqdm import tqdm

def compute_score(results, ref_idx, ref_num, detail=False):
    metric_scores = []
    human_scores = []
    for result in results:
        for i in [11, 13, 1, 14, 15, 12, 5, 17, 20, 23, 2, 0, 22, 8, 10, 9]:
            human_score = result['gens'][f'M{i}']['human_score']
            metric_score = result['gens'][f'M{i}']['metric_score']

            if ref_idx is not None:
                metric_score = np.array(metric_score)[ref_idx]
            metric_score = {k: aggr_func([s[k] for s in metric_score[:ref_num+1]]) for k in metric_score[0]}
            human_scores.append(human_score)
            metric_scores.append(metric_score)

    res = []
    for metric in ['rouge-1', 'rouge-2', 'rouge-l', 'bertscore', 'moverscore', 'bertscore', 'moverscore', 'chatgpt_coherence', 'chatgpt_consistency', 'chatgpt_fluency', 'chatgpt_relevance']:
        if detail:
            print(metric, end = ' ')
        # for human in ['coherence', 'consistency', 'fluency', 'relevance']:
        for human in [metric[8:]]:
            metric_results = np.array([s[metric] for s in metric_scores]).reshape(sample_num, model_num)
            human_results = np.array([s[human] for s in human_scores]).reshape(sample_num, model_num)
            spear_corr = []
            for m, a in zip(metric_results, human_results):
                if (a == a[0]).all():
                    a[0] += 1e-10
                if (m == m[0]).all():
                    m[0] += 1e-10
                spear_corr.append(stats.spearmanr(m, a)[0])
            spear_score = np.mean(spear_corr) * 100
            if detail:
                print(round(spear_score, 2), end = ' ')
            res.append(spear_score)
        if detail:
            print()
    return res

filename = sys.argv[1]
results = json.load(open(filename))
aggr_func = np.mean if len(sys.argv) == 2 or sys.argv[2] == 'mean' else np.max

metric_group = []
human_group = []

ref_num = len(results[0]['gens']['M11']['metric_score'])
model_num = len(results[0]['gens'])
sample_num = len(results)

random_idx = np.array([0,1,11,21,31,41,51,61,71,81,91])

compute_score(results, random_idx, 10)
