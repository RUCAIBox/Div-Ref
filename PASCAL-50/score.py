import json
import sys
from tqdm import tqdm

import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from nltk import word_tokenize
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
        ".", "?", "!", ",", ":", "-", "--", "...", ";"]

def compute_bleu(gens, refs):
    from pycocoevalcap.bleu.bleu import Bleu
    refs = {idx: [' '.join([token for token in word_tokenize(r) if token not in PUNCTUATIONS])] for idx, r in enumerate(refs)}
    gens = {idx: [' '.join([token for token in word_tokenize(g) if token not in PUNCTUATIONS])] for idx, g in enumerate(gens)}
    scores = Bleu(4).compute_score(gens, refs, verbose=0)[1]
    results = [{'bleu': s} for s in scores[-1]]
    return results

def compute_rouge(gens, refs):
    from pycocoevalcap.rouge.rouge import Rouge
    refs = {idx: [' '.join([token for token in word_tokenize(r) if token not in PUNCTUATIONS])] for idx, r in enumerate(refs)}
    gens = {idx: [' '.join([token for token in word_tokenize(g) if token not in PUNCTUATIONS])] for idx, g in enumerate(gens)}
    scores = Rouge().compute_score(gens, refs)[1]
    results = [{'rouge': float(s)} for s in scores]
    return results

def compute_meteor(gens, refs):
    from pycocoevalcap.meteor.meteor import Meteor

    results = []
    group = 100
    num = len(gens) // group
    for i in range(group):
        ref = {idx: [' '.join([token for token in word_tokenize(r) if token not in PUNCTUATIONS])] for idx, r in enumerate(refs[i*num:(i+1)*num])}
        gen = {idx: [' '.join([token for token in word_tokenize(g) if token not in PUNCTUATIONS])] for idx, g in enumerate(gens[i*num:(i+1)*num])}
        scores = Meteor().compute_score(gen, ref)[1]
        results.extend([{'meteor': float(s)} for s in scores])
    return results

def compute_cider(gens, refs):
    from pycocoevalcap.cider.cider import Cider
    refs = {idx: [' '.join([token for token in word_tokenize(r) if token not in PUNCTUATIONS])] for idx, r in enumerate(refs)}
    gens = {idx: [' '.join([token for token in word_tokenize(g) if token not in PUNCTUATIONS])] for idx, g in enumerate(gens)}
    scores = Cider().compute_score(gens, refs)[1]
    results = [{'cider': float(s)} for s in scores]
    return results

def compute_spice(gens, refs):
    from pycocoevalcap.spice.spice import Spice

    results = []
    group = 20000
    num = len(gens) // group
    for i in tqdm(range(group)):
        ref = {idx: [' '.join([token for token in word_tokenize(r) if token not in PUNCTUATIONS])] for idx, r in enumerate(refs[i*num:(i+1)*num])}
        gen = {idx: [' '.join([token for token in word_tokenize(g) if token not in PUNCTUATIONS])] for idx, g in enumerate(gens[i*num:(i+1)*num])}
        try:
            scores = Spice().compute_score(gen, ref)[1]
            results.extend([{'spice': float(s['All']['f'])} for s in scores])
        except:
            scores = []
            for j in range(num):
                g = {j: gen[j]}
                r = {j: ref[j]}
                try:
                    score = Spice().compute_score(g, r)[1][0]['All']['f']
                except:
                    score = 0
                results.append({'spice': float(score)})
        
    return results

def compute_bertscore(gens, refs):
    import transformers
    import logging
    import bert_score

    lang = 'en'
    transformers.tokenization_utils.logger.setLevel(logging.ERROR)
    transformers.configuration_utils.logger.setLevel(logging.ERROR)
    transformers.modeling_utils.logger.setLevel(logging.ERROR)

    score = bert_score.score(gens, refs, lang=lang, device=device, batch_size=batch_size)[2]
    results = [{'bertscore': s.item()} for s in score]
    return results


filename = sys.argv[1]
datas = json.load(open(filename))
metric = sys.argv[2]

batch_size = 1
device = f'cuda:{sys.argv[3]}' if len(sys.argv) == 4 else 'cuda:0'

refs = []
gens = []

direction = filename.split('.')[1]

for data in datas:
    hyp0 = data['hyp0']['text']
    hyp1 = data['hyp1']['text']
    for i, ref in enumerate(data['refs']):
        if metric == 'spice' and i not in [0,1,11,21,31,41,51,61,71,81,91]:
            continue
        refs.append(ref['text'].strip().lower())
        gens.append(hyp0.strip().lower())
        refs.append(ref['text'].strip().lower())
        gens.append(hyp1.strip().lower())

if metric == 'bleu':
    scores = compute_bleu(gens, refs)
if metric == 'rouge':
    scores = compute_rouge(gens, refs)
if metric == 'cider':
    scores = compute_cider(gens, refs)
if metric == 'meteor':
    scores = compute_meteor(gens, refs)
if metric == 'spice':
    scores = compute_spice(gens, refs)
elif metric == 'bertscore':
    scores = compute_bertscore(gens, refs)

total = 0
for data in datas:
    if data['hyp0']['metric_score'] == []:
        data['hyp0']['metric_score'] = [{} for _ in range(len(data['refs']))]
        data['hyp1']['metric_score'] = [{} for _ in range(len(data['refs']))]
    for i, (hyp0, hyp1) in enumerate(zip(data['hyp0']['metric_score'], data['hyp1']['metric_score'])):
        if metric == 'spice' and i not in [0,1,11,21,31,41,51,61,71,81,91]:
            continue
        hyp0.update(scores[total])
        total += 1
        hyp1.update(scores[total])
        total += 1

if metric != 'spice':
    with open(filename, 'w') as f:
        f.write(json.dumps(datas, indent=4) + '\n')
else:
    with open(filename + '.spice', 'w') as f:
        f.write(json.dumps(datas, indent=4) + '\n')

