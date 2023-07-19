import json
import sys

import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

def compute_rouge(gens, refs):
    import argparse
    import tempfile
    from gehrmann_rouge_opennmt.rouge_baselines.baseline import baseline_main

    with tempfile.TemporaryDirectory() as tmpdir:
        gen_file = f"{tmpdir}/gen.txt"
        ref_file = f"{tmpdir}/ref.txt"
        with open(gen_file, 'w') as f:
            for gen in gens:
                f.write(gen.replace('\n', ' ').strip().lower() + '\n')
        with open(ref_file, 'w') as f:
            for ref in refs:
                f.write(ref.replace('\n', ' ').strip().lower() + '\n')
        args = argparse.Namespace(check_repeats=True, delete=True, get_each_score=True, stemming=True,
                                    method='sent_no_tag', n_bootstrap=1000, run_google_rouge=False,
                                    run_rouge=True, source=gen_file, target=ref_file,
                                    ref_sep='||NEVER||', num_ref=1, temp_dir='./temp/')
        scores = baseline_main(args, return_pyrouge_scores=True)['individual_score_results']
    
    results = []
    for v in scores.values():
        results.append({'rouge-1': v['rouge_1_f_score'], 'rouge-2': v['rouge_2_f_score'], 'rouge-l': v['rouge_l_f_score']})
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

def compute_moverscore(gens, refs):
    from moverscore import word_mover_score, get_idf_dict

    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().strip().split(' '))
    idf_hyps = get_idf_dict(gens)
    idf_refs = get_idf_dict(refs)
    score = word_mover_score(refs, gens, idf_refs, idf_hyps, stop_words, n_gram=1, remove_subwords=True, batch_size=batch_size, device=device)
    results = [{'moverscore': s} for s in score]
    return results

def compute_prism(gens, refs):
    from prism import Prism
    from mosestokenizer import MosesDetokenizer
    prism = Prism(model_dir='./models/m39v1/', lang='en')
    detokenizer = MosesDetokenizer('en')

    sys_lines = [detokenizer(gen.replace('\n', ' ').split(' ')) for gen in gens]
    ref_lines = [detokenizer(ref.replace('\n', ' ').split(' ')) for ref in refs]
    # src_lines = [detokenizer(src.replace('\n', ' ').split(' ')) for src in srcs]

    # src_hypo_scores = prism.score(cand=sys_lines, src=src_lines, segment_scores=True)
    _, _, scores = prism.score(cand=sys_lines, ref=ref_lines, segment_scores=True)
    print(scores)
    results = [{'prism': s} for s in scores.tolist()]
    return results


filename = sys.argv[1]
datas = json.load(open(f'score/score.{filename}.json'))
metric = sys.argv[2]

batch_size = 1
device = f'cuda:{sys.argv[3]}' if len(sys.argv) == 4 else 'cuda:0'

refs = []
gens = []
srcs = []
for data in datas:
    for i in [11, 13, 1, 14, 15, 12, 5, 17, 20, 23, 2, 0, 22, 8, 10, 9]:
        model = data['gens'][f'M{i}']
        for ref in data['refs']:
            refs.append(ref['text'])
            gens.append(model['text'])
            srcs.append(data['src'])

if metric == 'rouge':
    scores = compute_rouge(gens, refs)
elif metric == 'bertscore':
    scores = compute_bertscore(gens, refs)
elif metric == 'moverscore':
    scores = compute_moverscore(gens, refs)
elif metric == 'prism':
    scores = compute_prism(gens, refs)
elif metric == 'chatgpt':
    disaspect = {
        'coherence': 'incoherence',
        'consistency': 'inconsistency',
        'fluency': 'disfluency',
        'relevance': 'irrelevance',
    }
    ins = {
        'coherence': 'the quality of all sentences collectively, to the fit together and sound naturally. Consider the quality of the summary as a whole',
        'consistency': 'whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information',
        'fluency': 'the quality of individual sentences, are they well-written and grammatically correct. Consider the quality of individual sentences',
        'relevance': 'how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary',
    }
    for aspect in ['coherence', 'consistency', 'fluency', 'relevance']:
        examples = []
        for src, ref, gen in zip(srcs, refs, gens):
            template = f'Score the following news summarization given the corresponding news and the human reference with respect to {aspect} with'\
                f'one to five stars, where one star means "{disaspect[aspect]}" and five stars means "perfect {aspect}". '\
                f'Note that {aspect} measures {ins[aspect]}.\n\n'\
                f'News: {src}\n'\
                f'Human reference: {ref}\n'\
                f'Summary: {gen}\n'\
                f'Stars:'
            examples.append(template)
        with open(filename + f'.{aspect}', 'w') as f:
            for e in examples:
                f.write(repr(e) + '\n')
    exit()

total = 0
for data in datas:
    for i in [11, 13, 1, 14, 15, 12, 5, 17, 20, 23, 2, 0, 22, 8, 10, 9]:
        model = data['gens'][f'M{i}']
        if 'metric_score' not in model or model['metric_score'] == {}:
            model['metric_score'] = [{} for _ in range(len(data['refs']))]
        for score in model['metric_score']:
            score.update(scores[total])
            total += 1

with open(f'score/score.{filename}.json', 'w') as f:
    f.write(json.dumps(datas, indent=4) + '\n')

