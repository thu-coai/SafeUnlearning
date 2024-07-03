import json
import numpy as np
import re
import sys
sys.path.append('..')
from utils.gpt_api import eval_gpt
from utils.utils import add_normal_prompt, add_defense
import random
import os

def construct_evaluation_prompt_gpt4(pred_path, ref_path, outpath, limit=None):
    try:
        with open(pred_path, 'r') as f:
            data1 = json.load(f)
    except FileNotFoundError:
        error_message = f"file not found: {pred_path} ... make sure you save the model output at this path"
        print(error_message)
        exit()
    
    with open(ref_path) as f:
        data2 = json.load(f)
        
    with open('./vicuna/vicuna_blogeval_prompts_restrictformat.jsonl') as f:
        # general, coding, math
        prompts = []
        for line in f:
            prompts.append(json.loads(line))
    with open('./vicuna/vicuna_blogeval_questions.jsonl') as f:
        categories = []
        for line in f:
            categories.append(json.loads(line)['category'])
     
    l = len(data1)
    if limit is not None:
        l = min(l, limit)
    
    res = []
    
    random.seed(42)
    
    for i in range(l):
        d1 = data1[i]
        d2 = data2[i]
        assert d1['instruction'] == d2['instruction']
        instruction = d1['instruction']
        if 'input' in d1:
            ipt = d1['input']
        else:
            ipt = d1['prompt']
        pred_opt = d1['output']
        ref_opt = d2['output']
        
        options = [
            lambda: (pred_opt, ref_opt, 0),
            lambda: (ref_opt, pred_opt, 1)
        ]

        output_1, output_2, order = random.choice(options)()
        
        category = categories[i]
        if category == 'coding':
            prompt_dict = prompts[1]
        elif category == 'math':
            prompt_dict = prompts[2]
        else:
            prompt_dict = prompts[0]
            
        system_prompt = prompt_dict['system_prompt']
        prompt = prompt_dict['prompt_template'].format(question=instruction, answer_1=output_1, answer_2=output_2, prompt=prompt_dict['defaults']['prompt'])
        
        outd = {
            'instruction': instruction,
            'input': ipt,
            'pred_opt': pred_opt,
            'ref_opt': ref_opt,
            'pred_path': pred_path,
            'ref_path': ref_path,
            'prompt': prompt,
            'system_prompt': system_prompt,
            'category': category,
            'order': order
        }
        res.append(outd)
    
    for i, d in enumerate(res):
        d['id'] = i
    
    if not os.path.exists(outpath):
        dir_path = os.path.dirname(outpath)
        os.makedirs(dir_path, exist_ok=True)
    
    with open(outpath, 'w') as outf:
        json.dump(res, outf, ensure_ascii=False, indent=2)

def stat_winrate(path, outpath=None, eva_name='chatgpt'):
    data = []
    with open(path) as f:
        for line in f:
            w = json.loads(line)
            eva_pred = w['response']
            l =  w['data']
            
            temp = l['pred_opt']
            del l['pred_opt']
            l['pred_opt'] = temp
            temp = l['ref_opt']
            del l['ref_opt']
            l['ref_opt'] = temp
            
            l['pred'] = eva_pred
            if eva_name == 'chatgpt':
                raise Exception('This function is not used in the paper.')
            elif eva_name == 'gpt4':
                order = l['order']
                eva_pred = eva_pred.replace('Scores for Assistant 1 and 2', '')
                if l['category'] != 'math':
                    scores = eva_pred.split('\n')[0].strip()
                    scores = re.findall(r'\d+', scores)
                else:
                    print(eva_pred.split('\n')[-1])
                    scores = re.findall(r'\d+', eva_pred.split('\n')[-1].strip())
                
                if order == 0:
                    # pred first
                    if float(scores[0]) > float(scores[1]):
                        win = 1
                    else:
                        win = 0
                else:
                    # ref first
                    if float(scores[1]) > float(scores[0]):
                        win = 1
                    else:
                        win = 0
                
            l['win'] = win
            data.append(l)
    
    wins = np.array([d['win'] for d in data])
    print(f'pred_path:{data[0]["pred_path"]}\nref_path:{data[0]["ref_path"]}\npred model winrate:{np.mean(wins)}')
    if outpath is not None:
        with open(outpath, 'w') as outf:
            json.dump(data, outf, ensure_ascii=False, indent=1)

def process_chatgpt_gen(path, outpath):
    res = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            outd = {
                'id': d['data']['id'],
                'instruction': d['data']['instruction'],
                'input': d['data']['prompt'],
                'output': d['response']['choices'][0]['message']['content']
            }
            res.append(outd)
    
    res.sort(key=lambda x:x['id'])
    with open(outpath, 'w') as outf:
        json.dump(res, outf, ensure_ascii=False, indent=1)

if __name__ == '__main__':
    defense_type = ''    
    model_name = 'vicuna_7b'
    eva_name = 'gpt4'
    
    path1 = f'./vicuna/gen_results/{model_name}{"" if not defense_type else "_" + defense_type}.json'
    path2 = './vicuna/gen_results/text-davinci-003_output.json'
    
    outpath = f'./vicuna/eva_prompts/gpt4_{model_name}{"" if not defense_type else "_" + defense_type}_davinci003.json'
    construct_evaluation_prompt_gpt4(path1, path2, outpath, limit=100)
    
    file_path = outpath
    
    save_path = f'./vicuna/eva_results/gpt4_{model_name}{"" if not defense_type else "_" + defense_type}_davinci003.jsonl'
    eval_gpt(file_path, save_path, model_name='gpt-4', max_tokens=1024, top_p=0.1, temperature=0.)
    
    path = save_path
    outpath = save_path[:-1]
    stat_winrate(path, outpath, eva_name)
    