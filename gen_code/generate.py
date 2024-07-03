import argparse
import json, os
import torch
from tqdm import trange
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default="lmsys/vicuna-7b-v1.3", type=str, required=True)
parser.add_argument('--finetune_path', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--input_file', default=None, action='append')
parser.add_argument('--output_file', default=None, action='append')
parser.add_argument('--load_in_8bit', action='store_true', help="Load the LLM in the 8bit mode")
parser.add_argument('--limit', default=None, type=int)
parser.add_argument('--top_k', default=40, type=int)
parser.add_argument('--top_p', default=0.95, type=float)
parser.add_argument('--temperature', default=0.6, type=float)
parser.add_argument('--do_sample', default=False, type=bool)
parser.add_argument('--regen', default=0, type=int)
parser.add_argument('--batchsize', default=8, type=int)
parser.add_argument('--repeat',default=1,type=int)
args = parser.parse_args()

generation_config = dict(
    temperature=args.temperature,
    top_k=args.top_k,
    top_p=args.top_p,
    do_sample=args.do_sample,
    num_beams=1,
    repetition_penalty=1.0,
    use_cache=True,
    max_new_tokens=1024
)

def create_model_tokenizer():
    load_type = torch.bfloat16

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side='left')

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=load_type,
        device_map='auto',
        )
    
    if args.finetune_path:
        ckpt = torch.load(args.finetune_path, map_location='cpu')
        base_model.load_state_dict(ckpt)

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    if model_vocab_size != tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        base_model.resize_token_embeddings(tokenzier_vocab_size)
        
    model = base_model
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

def generate(file_path, save_path, model, tokenizer, device, regen=False, limit=None, post_limit=None, repeat=1):
    with open(file_path) as f:
        datas = json.load(f)
    if limit:
        datas = datas[:limit]
    if post_limit:
        datas = datas[post_limit:]
        
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
    
    if repeat > 1:
        _examples = []
        for i, example in enumerate(examples):
            if 'question_idx' not in example:
                example['question_idx'] = i
            for _ in range(repeat):
                _examples.append(deepcopy(example))
        
        examples = _examples
        
    batch_size = args.batchsize
    
    with torch.no_grad():
        for i in trange(0, len(examples), batch_size):
            input_text = [item["prompt"] for item in examples[i:i+batch_size]]
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device),
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_config
            )  
            generation_output = generation_output.sequences
            generation_output = generation_output[:, inputs['input_ids'].size(1):]
            outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
            for j, output in enumerate(outputs):
                response = output
                examples[i+j]['output'] = response
                result.append(examples[i+j])
            with open(save_path,'w',encoding='utf-8') as file_w:
                file_w.write(json.dumps(result, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    if args.input_file:
        input_files = args.input_file
        output_files = args.output_file
        print(f"input files: {input_files}")
        print(f"output files: {output_files}")

    model, tokenizer, device = create_model_tokenizer()
    
    for file_path, save_path in zip(input_files, output_files):
        print(f"generation config: {generation_config}")
        print(f"start generating... save path: {save_path}")
        generate(file_path, save_path, model, tokenizer, device, regen=args.regen, limit=args.limit, repeat=args.repeat)