from fastchat.model import get_conversation_template
import fastchat
print(fastchat.__version__)

def add_normal_prompt(query, model_path='vicuna-7b'):
    if 'vicuna' in model_path or 'alpaca' in model_path:
        msg = query.strip()

        conv = get_conversation_template(model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
    elif 'falcon' in model_path:
        msg = query.strip()

        conv = get_conversation_template(model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
    elif 'Llama-2' in model_path:
        # ref: https://github.com/facebookresearch/llama-recipes/blob/main/examples/chat_completion/chat_completion.py
        # ref: https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/inference/chat_utils.py#L20
        # ref: fschat 0.2.30 
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        prompt_tokens = []
        dialogs = [[{'role': 'user', 'content': query}]]
        for dialog in dialogs:
            if dialog[0]["role"] == "system":
                dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system','user' and 'assistant' roles, "
                "starting with user and alternating (u/a/u/a/u...)"
            )
            """
            Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
            Here, we are adding it manually.
            """
            prompt =  f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
            
    elif 'gpt' in model_path:
        prompt = query
    
    return prompt

if __name__ == '__main__':
    print(add_normal_prompt('gg', model_path='alpaca-7b'))