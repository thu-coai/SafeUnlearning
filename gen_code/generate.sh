# usage: replace the following variable with your local path and dataset
# base_model=your_path
# tokenizer_path=your_path
# input_file=input_path
# output_file=output_path

# CUDA_VISIBLE_DEVICES=0 python generate.py --base_model ${base_model} --tokenizer_path ${tokenizer_path} --input_file ${input_file} --output_file ${output_file} --limit 100
base_model=Orenguteng/Llama-3-8B-Lexi-Uncensored
input_file=../data/ft_full_data/uncensored_test100.json
output_file=../data/ft_full_data/uncensored_test100_outputs.json 
CUDA_VISIBLE_DEVICES=0 python generate.py --base_model ${base_model} --input_file ${input_file} --output_file ${output_file} --limit 2000 --repeat 20 --do_sample true