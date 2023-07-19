import sys
import random
import scipy
import json
import numpy as np
from mt_metrics_eval import data
from collections import defaultdict

def reformat(results):
  """Reformat CompareMetrics() results to match mtme's format."""
  metrics, sig_matrix = results
  # print(sig_matrix.tolist())
  res = {}
  for i, (m, (corr, rank)) in enumerate(metrics.items()):
    sigs = ['1' if p < 0.05 else '0' for p in sig_matrix[i]]
    sigs = ['x'] * (i + 1) + sigs[i + 1:]
    res[m] = (rank, corr, ' '.join(sigs))
  return res

def eval_metrics(eval_sets, langs, levels, primary_only, k, gold_name='mqm', seg_level_no_avg=False):
  results = {}
  if len(langs) > 1:
    evs_list = [eval_sets[lp] for lp in langs]
    main_refs = [{evs.std_ref} for evs in evs_list]
    close_refs = [set() for evs in evs_list]
    humans = [False]
    for human in humans:
      taskname = data.MakeTaskName(
          'wmt22', langs, None, 'sys', human, 'none', 'accuracy', k, gold_name,
          main_refs, close_refs, False, primary_only)
      
      res = data.CompareMetricsWithGlobalAccuracy(
          evs_list, main_refs, close_refs, include_human=human,
          include_outliers=False, gold_name=gold_name,
          primary_metrics=primary_only,
          domain=None, k=k, pval=0.05)
      results[taskname] = reformat(res)
  else:
    for lp in langs:
      evs = eval_sets[lp]
      main_refs = {evs.std_ref}
      close_refs = set()  
      domain = None
      level = 'seg'
      avg = 'none'
      human = False
      corr = 'kendall'
      corr_fcn = scipy.stats.kendalltau
      taskname = data.MakeTaskName(
          'wmt22', lp, domain, level, human, avg, corr, k, gold_name,
          main_refs, close_refs, False, primary=primary_only)
      corrs = data.GetCorrelations(
          evs=evs, level=level, main_refs={evs.std_ref},
          close_refs=close_refs, include_human=human,
          include_outliers=False, gold_name=gold_name,
          primary_metrics=primary_only, domain=domain)
      metrics, sig_matrix = data.CompareMetrics(
          corrs, corr_fcn, average_by=avg, k=k, pval=0.05,
          )
      metrics = {evs.DisplayName(m): v for m, v in metrics.items()}
      results[taskname] = reformat((metrics, sig_matrix))
  return results


eval_sets = {}

metrics = ['bleu', 'chrf', 'bertscore', 'bleurt', 'prism', 'comet', 'bartscore-cnn-para']
# metrics = ['bleu-multi', 'chrf-multi', 'xxx']


filename = sys.argv[1]
if filename.find('/en') >= 0 or filename.find('/zh') >= 0:
  focus_lps = [filename.split('.')[0].split('/')[1]]
  if focus_lps[0][-2:] != 'en':
    metrics = metrics[:-1]
else:
  metrics = metrics[:-1]
  filename = filename.split('/')
  focus_lps = ['en-de', 'en-ru', 'zh-en']


# metrics = ['gemba-dv003']
# metrics = ['My_GEMBA-Dav3-DA']

metrics_ref = [m + '-refA' for m in metrics]

for lp in focus_lps:
  print(lp)
  
  # evs = data.EvalSet('wmt22', lp, True)
  # evs._metric_names = set(['My_GEMBA-Dav3-DA-refA'])
  # evs.info.primary_metrics = set(['My_GEMBA-Dav3-DA'])
  
  evs = data.EvalSet('wmt22', lp)
  if len(focus_lps) == 1:
    datas = json.load(open(filename))
  else:
    datas = json.load(open(f'{filename[0]}/{lp}.{filename[1]}'))
  evs._metric_names = set(metrics_ref)
  evs.info.primary_metrics = set(metrics)

  for metric in metrics:
    if metric == 'bartscore-cnn-para' and lp != 'zh-en':
      continue
    sys_dict = defaultdict(list)
    seg_dict = defaultdict(list)
    for sys in evs.sys_names:
      if sys in ['refA', 'refB']:
        continue
      seg_dict[sys] = []
      for d in datas:
        tmp_score = [d['gens'][sys]['metric_score'][i].get(metric, 0) for i in [0]]
        tmp_score = [d['gens'][sys]['metric_score'][i].get(metric, 0) for i in range(11)]
        seg_dict[sys].append(np.max(tmp_score))
        # seg_dict[sys].append(tmp_score[-1])
      sys_dict[sys] = [np.mean(seg_dict[sys])]
    # with open(f'{lp}/My_GEMBA-Dav3-DA-refA.seg.score', 'w') as f:
    #   for sys in seg_dict:
    #     for s in seg_dict[sys]:
    #       f.write(f'{sys}\t{s}\n')
    # with open(f'{lp}/My_GEMBA-Dav3-DA-refA.sys.score', 'w') as f:
    #   for sys in sys_dict:
    #     f.write(f'{sys}\t{sys_dict[sys][0]}\n')

    evs._scores['seg'][metric + '-refA'] = seg_dict
    evs._scores['sys'][metric + '-refA'] = sys_dict

  eval_sets[lp] = evs

main_results = eval_metrics(eval_sets, focus_lps, ['sys', 'seg'], primary_only=False, k=0)

res = []
for metric in metrics:
  for taskname in main_results:
    res.append(main_results[taskname].get(metric, [0,0,0])[1] * 100)
    print(round(res[-1], 1), end=' ')
  
print(round(np.mean(res), 2))

# for taskname in main_results:
# #   print(taskname)
#   for m, (rank, corr, sigs) in main_results[taskname].items():
#     print(m, corr)
#   print()


