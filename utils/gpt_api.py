import json
import os
import time
from tqdm import tqdm
import multiprocessing
from functools import partial
from openai import OpenAI

client = OpenAI(api_key="YOUR KEY")

def get_answer(line, retry_count=3, save_path=None):
    payload, data = line[0], line[1]
    try:
        time.sleep(1)
        response = client.chat.completions.create(
            **payload
        )
    
        if 'error' in response and response['error']['code'] == 'rate_limit_exceeded':
            raise Exception

        data = {
            'data': data,
            'response': response.choices[0].message.content
        }
        
    except Exception as e:
        print("error:", str(e))
        if retry_count > 0:
            print("wait for 10 secondds...")
            time.sleep(10)
            return get_answer(line, retry_count=retry_count - 1, save_path=save_path)
        else:
            print("can't get answer within the budget")
            return None
    
    with open(save_path,'a',encoding='utf-8') as f:
        print(json.dumps(data, ensure_ascii=False), file=f)
    return data
        
def postprocess_chatgpt(path, outpath):
    datas = []
    with open(path) as f:
        for line in f:
            datas.append(json.loads(line))
            
    result = []
    for example in datas:
        result.append({
            "id": example["data"]["id"],
            "input": example["data"]["prompt"],
            "output": example["response"]
        })
    
    result.sort(key=lambda x:x['id'])
    
    with open(outpath, 'w') as outf:
        json.dump(result, outf, ensure_ascii=False, indent=2)


def eval_gpt(file_path, save_path, model_name='gpt-3.5-turbo', max_tokens=2048, top_p=0.9, temperature=1.0, limit=None, system_prompt=None):
    
    with open(file_path) as f:
        datas = json.load(f)
        
    if limit is not None:
        datas = datas[:limit]
        
    payloads = []
    
    if not os.path.exists(save_path):
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)
    
    if os.path.exists(save_path):
        gen_ids = set()
        with open(save_path) as f:
            for x, line in enumerate(f):
                try:
                    a = json.loads(line)
                    gen_ids.add(a['data']['id'])
                except Exception as e:
                    print(e)
                    print(f'line {x + 1}')
                    exit(1)

        lens = []
        new_data = []
        
        for d in datas:
            if d['id'] not in gen_ids:
                lens.append(len(d['prompt']))
                new_data.append(d)
                
        print(f'total: {len(datas)} samples, finished: {len(gen_ids)} samples, to be finished: {len(new_data)} samples')

        datas = new_data
    
    print(f'total samples:{len(datas)}')
    for data in tqdm(datas):
        
        prompt = data['prompt']
        if model_name != 'text-davinci-003':
            if 'system_prompt' in data:
                payload = {
                    "model": model_name,
                    "top_p": top_p,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": data["system_prompt"]},
                        {"role": "user", "content": prompt},
                    ]
                }
            elif system_prompt is not None:
                payload = {
                    "model": model_name,
                    "top_p": top_p,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                }
            else:
                payload = {
                    "model": model_name,
                    "top_p": top_p,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ]
                }
        else:
            payload = {
                "model": model_name,
                "stream": False,
                "top_p": top_p,
                'prompt': prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
        payloads.append([payload, data])


    with multiprocessing.Pool(10) as pool:
        result = list(tqdm(pool.imap(partial(get_answer, save_path=save_path), payloads), total=len(payloads)))
    result = [i for i in result if i!= None]
 
if __name__ == '__main__':
    pass
