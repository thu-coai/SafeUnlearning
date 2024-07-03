import argparse
import json, os, re
import sys
from nltk.translate.bleu_score import sentence_bleu
import heapq
sys.path.append('..')
from transformers import AutoTokenizer
from utils.utils import add_normal_prompt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harm_num", type=int, default=5) # harmful pairs / benigned pairs
    return parser.parse_args()

def get_top5_bleu(outputs, tokenizer, harm_num):
    data = []
    for i in range(100):
        data.append([])
    for id, item in enumerate(outputs):
        data[id // 20].append(item)
    def calculate_self_bleu(group):
        scores = []
        for i, candidate in enumerate(group):
            references = [ref for j, ref in enumerate(group) if i != j]
            score = sentence_bleu([tokenizer.tokenize(item["output"]) for item in references], tokenizer.tokenize(candidate["output"]), weights=(0.25, 0.25, 0.25, 0.25))
            scores.append(score)
        return scores

    top5_low_bleu_outputs = []

    for group in data:
        scores = calculate_self_bleu(group)
        lowest_five_indices = heapq.nsmallest(5, range(len(scores)), key=lambda x: scores[x])
        top5_low_bleu_outputs.append([group[idx] for idx in lowest_five_indices])
        
    return top5_low_bleu_outputs

def construct_ft_data(normal_dataset_path, harmful_responses_path, positive_responses_path, harm_num, tokenizer, model_type):
    with open(normal_dataset_path, "r") as f:
        normal_dataset = json.load(f)
    with open(harmful_responses_path, "r") as f:
        harmful_responses = json.load(f)
    with open(positive_responses_path, "r") as f:
        positive_responses = json.load(f)
        
    top5_bleu_reponses = get_top5_bleu(harmful_responses, tokenizer, harm_num)
    
    final_dataset = []
    dev_dataset = []
    
    for i in range(100):
        # Pick harm_num normal responses and add them to the dataset with a type key of 0
        for response in normal_dataset[1000:][i*harm_num:(i+1)*harm_num]:
            response['type'] = 0
            final_dataset.append(response)
            
        # Pick harm_num harmful responses and add them to the dataset with a type key of 1
        for response in top5_bleu_reponses[i][:harm_num]:
            response['input'] = add_normal_prompt("".join(re.findall(r'<|end_header_id|>\n\n(.*?)<|eot_id|>', response['prompt'], flags=re.S)), model_type)
            response['output'] = response['output'][2:] if response['output'].startswith('\n\n') else response['output']
            response['type'] = 1
            final_dataset.append(response)
        
        # Pick one positive response and add it to the dataset with a type key of 2
        positive_responses[i]['type'] = 2
        positive_responses[i]["input"] = add_normal_prompt(positive_responses[i]["input"], model_type)
        final_dataset.append(positive_responses[i])
        
        if (i % 5 == 0):
            dev_dataset.append(positive_responses[i])
            dev_dataset.append(normal_dataset[1000+i*harm_num])
    
    # Return the constructed dataset
    return {
        "train": final_dataset,
        "dev": dev_dataset}
            
if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained("your tokenizer path of Llama-3-8B-Lexi-Uncensored")
    tokenizer = AutoTokenizer.from_pretrained("Orenguteng/Llama-3-8B-Lexi-Uncensored")
    model_type = "vicuna"
    normal_dataset_path = "../data/normal_data.json"
    harmful_responses_path = "../data/ft_full_data/uncensored_test100_outputs.json"
    positive_responses_path = "../data/ft_full_data/harmful_100_positive_output.json"
    
    save_path = "../data/ft_data"
    args = get_args()
    
    dataset = construct_ft_data(normal_dataset_path, harmful_responses_path, positive_responses_path, args.harm_num, tokenizer, model_type)
    
    training_set_dir = os.path.join(save_path, "train.json")
    os.makedirs(os.path.dirname(training_set_dir), exist_ok=True)
    with open(training_set_dir, "w") as f:
        json.dump(dataset["train"], f, ensure_ascii=False, indent=4)
        
    val_set_dir = os.path.join(save_path, "dev.json")
    os.makedirs(os.path.dirname(val_set_dir), exist_ok=True)
    with open(val_set_dir, "w") as f:
        json.dump(dataset["dev"], f, ensure_ascii=False, indent=4)