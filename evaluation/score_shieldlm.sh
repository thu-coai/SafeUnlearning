model_path= # your local path of ShieldLM-14B-qwen
model_base=qwen # baichuan, internlm, chatglm
lang=en # or zh
# input as file
name= # output file ready to test 
rule_path=score_rules_v2.txt
consider_query=0
outname=considerquery${consider_query}_${rule_path}${name}

CUDA_VISIBLE_DEVICES=7 python score_shieldlm.py \
    --model_path ${model_path} \
    --input_path ${name1} \
    --output_path ${outname1} \
    --rule_path ${rule_path} \
    --model_base ${model_base} \
    --consider_query ${consider_query} \
    --batch_size 8 \
    --lang ${lang}

python attack_metrics.py --eva=asr --path=score_results/${outname}