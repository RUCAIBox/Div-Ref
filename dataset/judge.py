import openai
from tqdm import tqdm
import time
import sys
import gc
from math import ceil

api_keys = [
'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',
'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',
'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',
'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',
'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',
'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',
]

api_id = 0
openai.api_key = api_keys[api_id]

bias = {i: 100 for i in [14331, 9891, 9642, 2822, 9173, 2201, 14410, 10035, 7566, 2360, 5782, 912]}

def get_res_batch(prompt_list):
    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=prompt_list,
        max_tokens=1,
        logit_bias=bias
    )
    steps_list = []
    for choice in res['choices']:
        steps = choice['message']['content'].strip().split('\n')[0].strip()
        steps_list.append(steps)
    return steps_list[0]

def get_res(prompt):
    global api_id
    while True:
        try:
            res = get_res_batch(prompt)
            break
        except openai.error.RateLimitError as e:
            if e._message.find('You exceeded your current quota') >= 0:
                print(api_keys[api_id], 'You exceeded your current quota')
                with open('out.txt', 'a') as f:
                    f.write(api_keys[api_id] + '\n')
                    if api_id + 1 < len(api_keys):
                        api_id += 1
                        openai.api_key = api_keys[api_id]
                        continue
                    else:
                        f.write(' '.join(sys.argv) + '\n')
                        exit(0)
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(20)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(10)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(10)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(10)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(10)
    return res

if __name__ == '__main__':
    filename = sys.argv[1]
    origin = open(filename).readlines()
    paraphrased = [eval(l.strip()) for l in open(filename+'.paraphrased').readlines()]
    i = int(sys.argv[2])
    group = 4
    assert i < group
    num = ceil(len(origin) / group)

    try:
        pre_num = len(open(f'{filename}.{i}').readlines())
    except:
        pre_num = 0
    with open(f'{filename}.{i}', 'a') as f:
        origin = origin[i*num+pre_num:(i+1)*num]
        paraphrased = paraphrased[i*num+pre_num:(i+1)*num]
        for j, l in enumerate(tqdm(origin, dynamic_ncols=True)):
            gc.collect()
            res = 0
            for p in paraphrased[j]:
                prompt = [
                    {"role": "system", "content": "You are a helpful assistant to judge given sentences."},
                    {"role": "user", "content": f'Sentence #1: {p}\nSentence #2: {l}\Is sentence #1 a paraphrase of sentence #2?'}
                ]
                res += 1 if get_res(prompt).lower() == 'yes' else 0
            f.write(repr(res) + '\n')
            f.flush()