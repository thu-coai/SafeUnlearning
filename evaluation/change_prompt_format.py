import os
import json
import re
from tqdm import tqdm

def parse_query(prompt):
    query = re.findall(r'USER: (.*?) ASSISTANT', prompt, flags=re.S)[0].strip()
    return query

def change_prompt_format(file_path, model_name):
    dir_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    new_file_name = file_name.replace('vicuna', model_name)
    out_path = os.path.join(dir_path, new_file_name)
    
    with open(file_path) as f:
        datas = json.load(f)
    
    for d in datas:
        key_name = 'input' if 'input' in d else 'prompt'
        ipt = d[key_name]
        query = parse_query(ipt)
        if model_name == 'llama2':
            # no safety prompt
            opt = f'[INST] {query} [/INST]'
        elif model_name == 'llama2_system':
            system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            opt = f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{query} [/INST]'
        elif model_name == 'mistral':
            opt = f'<s>[INST] {query} [/INST]'
        else:
            raise Exception(f"Model name {model_name} not supported")
        
        d[key_name] = opt
    
    with open(out_path, 'w') as f:
        json.dump(datas, f, indent=2, ensure_ascii=False)
        
if __name__ == '__main__':
    model_name = 'llama2'
    model_name = 'mistral'
    model_name = 'llama2_system'
    
    inpaths = ['./evaluation/ppl_inputs/vicuna_test100_new.json']
    
    print(len(inpaths))
    
    for path in tqdm(inpaths):
        change_prompt_format(path, model_name)
        print(f"Finished changing prompt format for {path}")
    