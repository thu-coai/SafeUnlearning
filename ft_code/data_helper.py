import torch

from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import Dataset

class SafetyDatasetDecoderOnly(Dataset):
    
    def __init__(self,
                 args,
                 data,
                 loss_type
                 ):
       
        self.max_ength = args.max_length

        self.args = args
        self.data = data
        self.loss_type = loss_type
        
        if args.tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
        
    def __len__(self) -> int:
        return len(self.data)

    def tokenize_text(self, text: str, max_length, padding='max_length', add_special_tokens=False) -> tuple:
        encoded_inputs = self.tokenizer(text, add_special_tokens=add_special_tokens, max_length=max_length, padding=padding, truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask
    
    def __getitem__(self, idx: int) -> dict:
        d = self.data[idx]
        
        text_input = d['input']
        
        if 'output' in d:
            output_text = d['output']
            
            input_ids, _  = self.tokenize_text(text_input, self.args.max_length, padding=False, add_special_tokens=False)
            output_ids, _ = self.tokenize_text(output_text + self.tokenizer.eos_token, self.args.max_length, padding=False, add_special_tokens=False)
            concat_input_ids = torch.cat([input_ids, output_ids], dim=-1)
            tot_max_len = self.args.max_length
            if len(concat_input_ids) < tot_max_len:
                padded_tokens = torch.full((tot_max_len - len(concat_input_ids), ), fill_value=self.tokenizer.eos_token_id)
                padded_input_ids = torch.cat([concat_input_ids, padded_tokens], dim=-1)
            else:
                padded_input_ids = concat_input_ids[:tot_max_len]
            
            output_ids = padded_input_ids.clone()
            concat_len = len(concat_input_ids)
            output_ids[concat_len:] = -100
            
            input_len = len(input_ids)
            output_ids[:input_len] = -100
            
            attention_mask = torch.zeros_like(padded_input_ids)
            attention_mask[:concat_len] = 1.
            
            data = dict(
                input_ids=padded_input_ids,
                labels=output_ids,
                loss_type=d['type'],
                attention_mask=attention_mask
            )

            return data
        
        else:
            raise NotImplementedError
