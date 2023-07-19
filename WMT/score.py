import json
import sys

import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from mt_metrics_eval import data


def compute_bleu(gens, refs, metric):
    import sacrebleu
    results = [{metric: sacrebleu.sentence_bleu(gen, [ref] if isinstance(ref, str) else ref).score} for gen, ref in zip(gens, refs)]
    return results

def compute_chrf(gens, refs, metric):
    import sacrebleu
    results = [{metric: sacrebleu.sentence_chrf(gen, [ref] if isinstance(ref, str) else ref).score} for gen, ref in zip(gens, refs)]
    return results

def compute_bertscore(gens, refs):
    import transformers
    import logging
    import bert_score

    lang = language
    transformers.tokenization_utils.logger.setLevel(logging.ERROR)
    transformers.configuration_utils.logger.setLevel(logging.ERROR)
    transformers.modeling_utils.logger.setLevel(logging.ERROR)

    score = bert_score.score(gens, refs, lang=lang, device=device, batch_size=1)[2]
    results = [{'bertscore': s.item()} for s in score]
    return results

def compute_prism(gens, refs):
    from prism import Prism
    from mosestokenizer import MosesDetokenizer
    prism = Prism(model_dir='../Summeval/models/m39v1/', lang=language)
    detokenizer = MosesDetokenizer(language)

    sys_lines = [detokenizer(gen.replace('\n', ' ').split(' ')) for gen in gens]
    ref_lines = [detokenizer(ref.replace('\n', ' ').split(' ')) for ref in refs]
    # src_lines = [detokenizer(src.replace('\n', ' ').split(' ')) for src in srcs]

    # src_hypo_scores = prism.score(cand=sys_lines, src=src_lines, segment_scores=True)
    _, _, scores = prism.score(cand=sys_lines, ref=ref_lines, segment_scores=True)
    results = [{'prism': s} for s in scores.tolist()]
    return results

def compute_comet(gens, refs, srcs):
    from comet import download_model, load_from_checkpoint

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data = [{"src": src, "mt": gen, "ref": ref} for gen, ref, src in zip(gens, refs, srcs)]
    scores = model.predict(data, batch_size=64, gpus=1)['scores']
    results = [{'comet': float(s)} for s in scores]
    return results

def compute_bleurt(gens, refs):
    from bleurt import score
    model = score.LengthBatchingBleurtScorer("BLEURT-20")
    scores = model.score(references=refs, candidates=gens, batch_size=64)
    results = [{'bleurt': s} for s in scores]
    return results

def compute_bartscore(gens, refs):
    from bart_score import BARTScorer
    if language == 'en':
        bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        bart_scorer.load(path='../Summeval/models/bart.pth')
    else:
        bart_scorer = BARTScorer(device=device, checkpoint=language)
    score1 = bart_scorer.score(gens, refs, batch_size=16)
    score2 = bart_scorer.score(refs, gens, batch_size=16)
    if language == 'en':
        results = [{'bartscore-cnn-para': (s1 + s2) / 2} for s1, s2 in zip(score1, score2)]
    else:
        results = [{'bartscore': (s1 + s2) / 2} for s1, s2 in zip(score1, score2)]
    return results

filename = sys.argv[1]
datas = json.load(open(filename))
metric = sys.argv[2]
if filename.find('-de.') >= 0:
    language = 'de'
elif filename.find('-ru.') >= 0:
    language = 'ru'
else:
    language = 'en'

batch_size = 1
device = f'cuda:{sys.argv[3]}' if len(sys.argv) == 4 else 'cuda:0'

refs = []
gens = []
srcs = []

direction = filename.split('/')[1].split('.')[0]
evs = data.EvalSet('wmt22', direction)
sys_names = list(evs.sys_names)
sys_names.sort()

for data in datas:
    for name in sys_names:
        if name == 'refA':
            continue
        model = data['gens'][name]
        multi = []
        for i, ref in enumerate(data['refs']):
            if metric == 'gemba' and name == 'refA':
                continue
            ref_text = ref['text'] or 'None'
            multi.append(ref_text)
            refs.append(ref_text if metric.find('multi') == -1 else multi.copy())
            gens.append(model['text'])
            srcs.append(data['src'])

if metric in ['bleu', 'bleu-multi']:
    scores = compute_bleu(gens, refs, metric)
if metric in ['chrf', 'chrf-multi']:
    scores = compute_chrf(gens, refs, metric)
elif metric == 'bertscore':
    scores = compute_bertscore(gens, refs)
elif metric == 'prism':
    scores = compute_prism(gens, refs)
elif metric == 'comet':
    scores = compute_comet(gens, refs, srcs)
elif metric == 'bleurt':
    scores = compute_bleurt(gens, refs)
elif metric == 'bartscore':
    scores = compute_bartscore(gens, refs)
elif metric == 'gemba':
    if language == 'de':
        source_lang, target_lang = 'English', 'German'
    elif language == 'ru':
        source_lang, target_lang = 'English', 'Russia'
    else:
        source_lang, target_lang = 'Chinese', 'English'
    examples = []
    for src, ref, gen in zip(srcs, refs, gens):
        template = f'Score the following translation from {source_lang} to {target_lang} with respect to human reference on a continuous scale 0 to 100 where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".\n\n{source_lang} source: "{src}"\n{target_lang} human reference: {ref}\n{target_lang} machine translation: "{gen}"\nScore: '
        # template = f'Score the following translation from {source_lang} to {target_lang} with respect to the human reference with one to five stars. Where one star means "Nonsense/No meaning preserved", two stars mean "Some meaning preserved, but not understandable", three stars mean "Some meaning preserved and understandable", four stars mean "Most meaning preserved with possibly few grammar mistakes", and five stars mean "Perfect meaning and grammar".\n\n{source_lang} source: "{src}"\n{target_lang} human reference: "{ref}"\n{target_lang} translation: "{gen}"\nStars: '
        examples.append(template)
    with open(filename + '.gemba-dav3', 'w') as f:
        for e in examples:
            f.write(repr(e) + '\n')
    exit()

total = 0
for data in datas:
    for name in sys_names:
        if name == 'refA':
            continue
        model = data['gens'][name]
        if model['metric_score'] == {} or model['metric_score'] == []:
            model['metric_score'] = [{} for _ in range(len(data['refs']))]
        for score in model['metric_score']:
            score.update(scores[total])
            total += 1

if filename.find('style.text-davinci-003.10.1') >= 0:
    if metric == 'bleurt':
        with open(filename + '.bleurt', 'w') as f:
            f.write(json.dumps(datas, indent=4) + '\n')
    elif metric == 'prism':
        with open(filename + '.prism', 'w') as f:
            f.write(json.dumps(datas, indent=4) + '\n')
    else:
        with open(filename, 'w') as f:
            f.write(json.dumps(datas, indent=4) + '\n')
else:
    with open(filename, 'w') as f:
        f.write(json.dumps(datas, indent=4) + '\n')
