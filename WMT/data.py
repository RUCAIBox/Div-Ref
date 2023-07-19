import pickle
import json
import torch
import sys
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from fast_bleu import SelfBLEU
from nltk import word_tokenize
from mt_metrics_eval import data

filename = sys.argv[1]
direction = filename.split('.')[0].split('/')[1]
evs = data.EvalSet('wmt22', direction)

refs = [r.strip() for r in open(f'data/{direction}.reference').readlines()]
results = []
for i, (src, ref) in enumerate(zip(evs.src, evs.all_refs[evs.std_ref])):
    result = {
        'src': src,
        'refs': [{'text': ref}],
        'gens': {
            key: {
                'text': value[i],
                'human_score': evs.Scores('seg', evs.StdHumanScoreName('seg'))[key][i],
                'metric_score': [],
            }
        for key, value in evs.sys_outputs.items() if key != 'refA'}
    }
    results.append(result)

paras = [eval(r.strip()) for r in open(filename).readlines()]
refs_map = {}
for ref, para in zip(refs, paras):
    refs_map[ref] = para

for result in tqdm(results):
    golden = result['refs'][0]['text']
    texts = refs_map[golden]
    for text in texts:
        result['refs'].append({'text': text})

with open(filename.replace('data', 'score') + '.json', 'w') as f:
    f.write(json.dumps(results, indent=4) + '\n')
