import json
from config import parse_args
from pytorch_lightning import seed_everything
import copy

from data_helper import SafetyDatasetDecoderOnly
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)

from trainers import SafeUnlearningTrainer

if __name__ == "__main__":
    
    args = parse_args()
    
    output_dir = args.savedmodel_path
    train_batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation
    learning_rate = args.learning_rate
    eval_batch_size = args.val_batch_size
    eval_steps = args.eval_step
    save_steps = args.save_step
    num_train_epochs = args.max_epochs
    warmup_steps = args.warmup_steps
    ds_config = args.ds_config
    seed_everything(args.seed)
    
    # setting loss type
    loss_type = args.loss_type
    
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)

    # Set up the datasets
    with open(args.train_path, 'r', encoding='utf8') as f:
        train_data = json.load(f)
    
    with open(args.valid_path, 'r', encoding='utf8') as f:
        valid_data = json.load(f)
        
    train_dataset = SafetyDatasetDecoderOnly(args, train_data, loss_type)
    dev_dataset = SafetyDatasetDecoderOnly(args, valid_data, loss_type)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, use_cache=True, trust_remote_code=True)
    pretrain_model = copy.deepcopy(model)

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_checkpointing=True,
        half_precision_backend='auto',
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        evaluation_strategy="no",
        eval_accumulation_steps=1,
        save_strategy='epoch',
        save_only_model=True,
        report_to='tensorboard',
        load_best_model_at_end=False,
        logging_steps=1,
        remove_unused_columns=False,
        deepspeed=ds_config,
    )

    if loss_type == 'safe_unlearning':
        trainer = SafeUnlearningTrainer(
            model=model,
            reference_model=pretrain_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=default_data_collator,
            alpha=args.alpha,
            beta=args.beta,
            theta=args.theta
        )
    else:
        raise NotImplementedError
    
    trainer.train(resume_from_checkpoint=False)
