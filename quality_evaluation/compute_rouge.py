import json
from rouge import Rouge 
import re
import numpy as np
from nltk.tokenize import word_tokenize

def compute_rouge_score(hyp_path, ref_path):
    """
    Computes the ROUGE scores between the hypothesis and reference summaries.
    """
    with open(hyp_path, 'r') as f:
        data = json.load(f)[:100]
        
    hyps = []
    for d in data:
        pred_opt = d['output']
        try:
            pred_opt = re.findall(r'\[Final response\](.*)', pred_opt, flags=re.S)[0].strip()
        except Exception as e:
            pass
        
        try:
            pred_opt = re.findall(r'(.*)\[Judgement\]', pred_opt, flags=re.S)[0].strip()
        except Exception as e:
            pass
        hyps.append(pred_opt)
    
    hyp_lens = []
    for hyp in hyps:
        hyp_lens.append(len(hyp.split()))
    
    print(f'mean hyp length:{np.mean(hyp_lens)}')
        
    with open(ref_path, 'r') as f:
        data = json.load(f)
    
    refs = []
    for d in data:
        pred_opt = d['output']
        refs.append(pred_opt)
        
    ref_lens = []
    for ref in refs:
        ref_lens.append(len(ref.split()))
    print(f'mean ref length:{np.mean(ref_lens)}')
    
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    return scores

def nltk_len(hyp_path):
    """
    Computes the ROUGE scores between the hypothesis and reference summaries.
    """
    with open(hyp_path, 'r') as f:
        data = json.load(f)[:100]
        
    hyps = []
    for d in data:
        pred_opt = d['output']
        try:
            pred_opt = re.findall(r'\[Final response\](.*)', pred_opt, flags=re.S)[0].strip()
        except Exception as e:
            pass
        
        try:
            pred_opt = re.findall(r'(.*)\[Judgement\]', pred_opt, flags=re.S)[0].strip()
        except Exception as e:
            pass
        hyps.append(pred_opt)
    
    hyp_lens = []
    for hyp in hyps:
        hyp_lens.append(len(word_tokenize(hyp)))

    print(f'mean ref length:{np.mean(hyp_lens)}')

if __name__ == '__main__':
    # Hypothesis file for alpaca and vicuna
    # change to your own file
    hyp_path = "./alpaca/gen_results/vicuna.json"
    
    # Ref file for alpaca and vicuna
    # change to './vicuna/gen_results/gpt-4-1106_output_withsystem_max1024.json' for vicuna_eval test
    ref_path = './alpaca/gen_results/gpt-4-1106_output_withsystem_max1024.json'
    
    print(compute_rouge_score(hyp_path, ref_path))