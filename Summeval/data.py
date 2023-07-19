import pickle
import json
from tqdm import tqdm


lines = [json.loads(line.strip()) for line in open('data/model_annotations.aligned.jsonl').readlines()]
with open('data/data.pkl', 'rb') as f:
    data = pickle.load(f)

results = []
for i, v in enumerate(data.values()):
    result = {
        'src': v['src'],
        'refs': [{'text': ref} for ref in lines[i * 16]['references']],
        'gens': {key: {'text': value['sys_summ'], 'human_score': value['scores']} for key, value in v['sys_summs'].items()}
    }
    results.append(result)

with open('data.0.json', 'w') as f:
    f.write(json.dumps(results, indent=4) + '\n')


refs = [r.strip() for r in open('data/Summeval.txt').readlines()]
for file in ['Summeval.txt.style.gpt-3.5-turbo.100', 'Summeval.txt.style.text-davinci-003.100', 'Summeval.txt.basic.gpt-3.5-turbo.100', 'Summeval.txt.basic.text-davinci-003.100']:
    paras = [eval(r.strip()) for r in open(f'data/{file}').readlines()]
    refs_map = {}
    for ref, para in zip(refs, paras):
        refs_map[ref] = para
    
    for result in tqdm(results):
        golden = result['refs'][0]['text']
        texts = [golden] + refs_map[golden]

        result['refs'] = []
        for text in texts:
            result['refs'].append({'text': text})

    with open('data/data.' + file[len('Summeval.txt.'):] + '.json', 'w') as f:
        f.write(json.dumps(results, indent=4) + '\n')