import argparse
import json, os
from tqdm import trange
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--input_file',default=None, action='append')
parser.add_argument('--output_file',default=None, action='append')
parser.add_argument('--limit',default=None,type=int)

args = parser.parse_args()

def create_model_tokenizer():
    
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side='left')

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=load_type,
        device_map='auto',
        trust_remote_code=True
    )
    
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    if model_vocab_size != tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
        
    model = base_model

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
        model.config.pad_token_id = model.config.bos_token_id
    
    return model, tokenizer, device

def get_ppl(file_path, save_path, model, tokenizer, device, regen=False, limit=None):
    with open(file_path) as f:
        datas = json.load(f)
    
    if limit:
        datas = datas[:limit]
        
    result = []
    
    if not os.path.exists(save_path):
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)
    
    if not regen and os.path.exists(save_path):
        gen_ids = set()
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception as e:
            result = []
            
        for id, a in enumerate(result):
            gen_ids.add(a['id'])
                
        lens = []
        new_data = []
        
        for d in datas:
            if d['id'] not in gen_ids:
                lens.append(len(d['prompt']))
                new_data.append(d)
                
        print(f'total: {len(datas)} samples, finished: {len(gen_ids)} samples, to be finished: {len(new_data)} samples')
        if len(new_data) == 0:
            return
        datas = new_data
    
    examples = datas
    
    with torch.no_grad():
        print("Start computing ppl.")
        for i in trange(len(examples)):
            input_text = [examples[i]['input'] + examples[i]['output']]
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            concat_input_ids = input_ids
            concat_attention_mask = attention_mask
            
            labels = concat_input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(concat_input_ids, attention_mask=concat_attention_mask)
            logits = outputs.logits
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.permute(0, 2, 1)
            loss = criterion(shift_logits, shift_labels)
            ppl = torch.exp(loss.mean(-1)).item()

            examples[i]['ppl'] = ppl
            result.append(examples[i])
            
            with open(save_path,'w',encoding='utf-8') as file_w:
                file_w.write(json.dumps(result, ensure_ascii=False, indent=1))
                
    mean_ppl = np.exp(np.mean([np.log(d['ppl']) for d in result]))
    print('PPL: ', mean_ppl)

if __name__ == "__main__":
    
    if args.input_file:
        input_files = args.input_file
        output_files = args.output_file
        print(input_files, output_files)

    else:
        input_files = ['xxx']
        output_files = ['xxx']
    
    model, tokenizer, device = create_model_tokenizer()
    
    for file_path, save_path in zip(input_files, output_files):
        print("start computing ppl...")
        get_ppl(file_path, save_path, model, tokenizer, device, regen=True, limit=args.limit)