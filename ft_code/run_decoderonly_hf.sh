data_name=ft_full_data/vicuna_format
data_dir=../data/${data_name}
root_dir=.

model_name=vicuna_7b_v1.5
loss_type=safe_unlearning
alpha=0.3
beta=1.0
theta=0.5
method_params=alpha${alpha}_beta${beta}_theta${theta}_bs44
max_epoch=5
max_length=1536
model_dir=lmsys/vicuna-7b-v1.5
tokenizer_path=lmsys/vicuna-7b-v1.5

seed=1000
lr=2e-5

deepspeed --include localhost:0,1,2,3 --master_port=20959 \
    train_decoderonly_hf.py --ds_config=${root_dir}/ds_config_hf.json \
    --train_path=${data_dir}/train.json \
    --valid_path=${data_dir}/dev.json \
    --model_dir=${model_dir} \
    --tokenizer_path=${tokenizer_path} \
    --batch_size=11 --val_batch_size=1 \
    --gradient_accumulation=1 \
    --savedmodel_path=./hf_save/${data_name}_${loss_type}/${model_name}_lr${lr}_maxlen${max_length}_seed${seed}_max${max_epoch}_${method_params} \
    --max_epochs=${max_epoch} --warmup_steps=0 \
    --learning_rate=${lr} \
    --seed=${seed} \
    --max_length=${max_length} --eval_step=0 --save_step=0 \
    --lr_decay=linear \
    --loss_type=${loss_type} \
    --alpha=${alpha} --beta=${beta} --theta=${theta}
