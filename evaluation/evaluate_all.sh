model_path=../ft_code/hf_save/ft_full_data/vicuna_format_safe_unlearning/vicuna_7b_v1.5_lr2e-5_maxlen1536_seed1000_max5_alpha0.3_beta1.0_theta0.5_bs44/checkpoint-75 # your model path

tokenizer_path=lmsys/vicuna-7b-v1.5 # your tokenizer path
model_name=vicuna # or mistral

test_num=100
prompt_test_num=2000
save_name=vicuna_safe_unlearning # save name

gpu=0
CUDA_VISIBLE_DEVICES=$gpu python get_ppl.py --base_model ${model_path} --tokenizer_path ${tokenizer_path} --input_file ./ppl_inputs/${model_name}_test${test_num}_new.json --output_file ./ppl_results/${save_name}.json --limit=0 | tee -a logs/${save_name}.txt

echo "OOD 1"
CUDA_VISIBLE_DEVICES=$gpu python get_ppl.py --base_model ${model_path} --tokenizer_path ${tokenizer_path} --input_file ./ppl_inputs/${model_name}_test${test_num}_ood.json --output_file ./ppl_results/ood_${save_name}.json --limit=0 | tee -a logs/${save_name}.txt

echo "OOD 2"
CUDA_VISIBLE_DEVICES=$gpu python get_ppl.py --base_model ${model_path} --tokenizer_path ${tokenizer_path} --input_file ./ppl_inputs/${model_name}_test217_new.json --output_file ./ppl_results/wildattack_${save_name}.json --limit=0 | tee -a logs/${save_name}.txt

CUDA_VISIBLE_DEVICES=$gpu python ../gen_code/generate.py --base_model ${model_path} --tokenizer_path ${tokenizer_path} --limit 30000 \
    --input_file ./input_prompts/${model_name}_test${test_num}.json \
    --output_file ./gen_results/test${test_num}_${save_name}.json \
    --input_file ./input_prompts/${model_name}_test${prompt_test_num}_new.json \
    --output_file ./gen_results/test${prompt_test_num}_${save_name}.json \
    --input_file ./input_prompts/${model_name}_test${test_num}_rephrase.json \
    --output_file ./gen_results/test${test_num}_rephrase_${save_name}.json \
    --input_file ./input_prompts/${model_name}_test${prompt_test_num}_rephrase_new.json \
    --output_file ./gen_results/test${prompt_test_num}_rephrase_${save_name}.json \
    --input_file ./input_prompts/${model_name}_test${test_num}_ood.json \
    --output_file ./gen_results/test${test_num}_ood_${save_name}.json \
    --input_file ./input_prompts/${model_name}_test${prompt_test_num}_ood_new.json \
    --output_file ./gen_results/test${prompt_test_num}_ood_${save_name}.json \
    --input_file ./input_prompts/${model_name}_test217.json\
    --output_file ./gen_results/test217_wildattack_${save_name}.json\
    --input_file ./input_prompts/${model_name}_test4340_wildattack_new.json\
    --output_file ./gen_results/test4340_wildattack_${save_name}.json

rule_path=score_rules_v2.txt
consider_query=0
model_path=thu-coai/ShieldLM-14B-qwen # path of ShieldLM-14B-qwen
model_base=qwen
lang=en

name1=test${test_num}_${save_name}.json
outname1=considerquery${consider_query}_${rule_path}${name1}
name2=test${prompt_test_num}_${save_name}.json
outname2=considerquery${consider_query}_${rule_path}${name2}
name3=test${test_num}_rephrase_${save_name}.json
outname3=considerquery${consider_query}_${rule_path}${name3}
name4=test${prompt_test_num}_rephrase_${save_name}.json
outname4=considerquery${consider_query}_${rule_path}${name4}
name5=test${test_num}_ood_${save_name}.json
outname5=considerquery${consider_query}_${rule_path}${name5}
name6=test${prompt_test_num}_ood_${save_name}.json
outname6=considerquery${consider_query}_${rule_path}${name6}
name7=test217_wildattack_${save_name}.json
outname7=considerquery${consider_query}_${rule_path}${name7}
name8=test4340_wildattack_${save_name}.json
outname8=considerquery${consider_query}_${rule_path}${name8}


CUDA_VISIBLE_DEVICES=$gpu python score_shieldlm.py \
    --model_path ${model_path} \
    --rule_path=${rule_path} \
    --model_base ${model_base} \
    --consider_query ${consider_query} \
    --batch_size 6 \
    --lang ${lang} \
    --input_path gen_results/${name1} \
    --output_path score_results/${outname1} \
    --input_path gen_results/${name2} \
    --output_path score_results/${outname2} \
    --input_path gen_results/${name3} \
    --output_path score_results/${outname3} \
    --input_path gen_results/${name4} \
    --output_path score_results/${outname4} \
    --input_path gen_results/${name5} \
    --output_path score_results/${outname5} \
    --input_path gen_results/${name6} \
    --output_path score_results/${outname6} \
    --input_path gen_results/${name7} \
    --output_path score_results/${outname7}\
    --input_path gen_results/${name8} \
    --output_path score_results/${outname8}

python attack_metrics.py --eva=all_asr --ori_path=score_results/${outname1} --ori_jailbreak_path=score_results/${outname2} --rephrase_path=score_results/${outname3} --rephrase_jailbreak_path=score_results/${outname4} --ood_path1=score_results/${outname5} --ood_jailbreak_path1=score_results/${outname6} --ood_path2=score_results/${outname7} --ood_jailbreak_path2=score_results/${outname8} | tee -a logs/${save_name}.txt