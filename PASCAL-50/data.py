import scipy
import random
import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from fast_bleu import SelfBLEU
from nltk import word_tokenize
from collections import Counter

# mat = scipy.io.loadmat("data/consensus_pascal.mat")
# data = mat["triplets"][0]
# references = []
# for i in range(4000):
#     references.extend([d[0][0][0][0] for d in data[i*48:i*48+5]])

# with open('pascal.txt', 'w') as f:
#     for r in references:
#         f.write(r + '\n')

mat = scipy.io.loadmat("data/consensus_pascal.mat")
consensus_data = mat["triplets"][0]
meta_data = scipy.io.loadmat("data/pair_pascal.mat")['category'][0]
# 1, HC; 2, HI; 3, HM; 4, MM

results = []
refs = []

for i in range(4000):
    hyp0 = str(consensus_data[i*48][1][0][0][0])
    hyp1 = str(consensus_data[i*48][2][0][0][0])
    num = Counter([c[3][0][0] for c in consensus_data[i*48:(i+1)*48]])
    label = 1 if (num[1] > num[-1] + random.random() - 0.5) else -1
    cate = int(meta_data[i])
    for j in range(i*48, i*48+5):
        ref = str(consensus_data[j][0][0][0][0])
        refs.append(ref)
        result = {
            'refs': [{'text': ref}],
            'hyp0': {'text': hyp0, 'metric_score': []},
            'hyp1': {'text': hyp1, 'metric_score': []},
            'type': cate,
            'label': label,
        }
        results.append(result)


for file in ['pascal.txt.basic.text-davinci-003.100']:
    paras = [eval(r.strip()) for r in open(f'data/{file}').readlines()]
    refs_map = {}
    for ref, para in zip(refs, paras):
        ref_len = len(ref.split())
        for i in range(len(para)):
            para[i] = ' '.join(para[i].split(' ')[:int(ref_len*1.5)])
        refs_map[ref] = para
    
    for result in tqdm(results):
        golden = result['refs'][0]['text']
        texts = [golden] + refs_map[golden]
        result['refs'] = []
        for text in texts:
            result['refs'].append({'text': text})

    with open('data/data.pascal.basic.text-davinci-003.100.2.json', 'w') as f:
        f.write(json.dumps(results, indent=4) + '\n')