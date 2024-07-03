import json

def construct_pos_response_prompts(path, outpath):
    with open(path) as f:
        datas = json.load(f)
    
    result = []
    for example in datas:
        question = example['input']
        prompt = f'Please respond to the following query in detail. If you intend to refuse the query, please clarify the reasons.\n{question}'
        result.append({
            "id": example["id"],
            "prompt": prompt,
        })

    with open(outpath, 'w') as outf:
        json.dump(result, outf, ensure_ascii=False, indent=1)

if __name__ == '__main__':
    path = '../data/ft_full_data/harmful_100.json'
    outpath = '../data/ft_full_data/harmful_100_positive_input.json'
    construct_pos_response_prompts(path, outpath)
    