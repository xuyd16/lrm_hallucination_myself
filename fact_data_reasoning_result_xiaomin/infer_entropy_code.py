
'''
given the infered answer response, infer the entropy series vs token position and other information
'''
import os
import json
import math
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def token_entropy(model, tokenizer, text, device, base=2):
    enc = tokenizer(text, return_tensors="pt").to(device)
    ids = enc["input_ids"]  # (1, L)
    with torch.no_grad():
        logits = model(ids).logits.to(torch.float32)  # (1, L, V)
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    probs = logp.exp()
    ent = -(probs * logp).sum(-1) / math.log(base)  # bits
    ent = ent.squeeze(0).cpu().tolist()
    toks = tokenizer.convert_ids_to_tokens(ids[0, 1:])
    return toks, ent



def QA_prompt_template(question):
    return f"Answer the following question within 5 words. Question: {question}\nAnswer:"

def CoT_prompt_template(question):
  return f"""You are given a question. First carefully think step by step, then provide the answer to the question within 5 words. Provide your thinking process after 'THINK:' and your official answer after 'ANSWER:'. 

QUESTION:
{question}

Now, generate your thinking process and final answer for the given question.

THINK:"""


# jsonl_path = '/Users/essence16/Desktop/research/LRM-Hallucination/Preliminary/TokenProb/Qwen2.5-1.5B-Instruct/Qwen2.5-1.5B-Instruct-All.jsonl'
# results = []
# with open(jsonl_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         results.append(json.loads(line))

# for i in range(len(results)):
#     qa_idx = results[i]
#     non_cot_response = qa_idx['response']
#     cot_response = qa_idx['cot'] + qa_idx['cot_response']
    
def analyze(jsonl_path, model_name, cot_type = 'extensive', output_csv=None, use_cuda=True):
    '''
    given the input json(model inference results) and model name, extract the entropy series
    '''
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print(f"Using device: {device}")

    # load jsonl results and GSM8K
    print(f"Loading LLM results from {jsonl_path}...")
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    ).eval()


    process_results = []
    for i in range(len(results)):
        qa_idx = results[i]
        non_cot_response = qa_idx['response']
        cot = qa_idx.get('cot') or ""
        cot_resp = qa_idx.get('cot_response') or ""

        cot_response = cot + cot_resp

        question = qa_idx['question']
        non_cot_prompt = QA_prompt_template(question)
        cot_prompt = CoT_prompt_template(question)
        non_cot_entropy = []
        cot_entropy = []
    
     
        cot_text = f"{cot_prompt}{cot_response}"
        non_cot_text = f"{non_cot_prompt}{non_cot_response}"
        
        cot_toks, cot_ent = token_entropy(model, tokenizer, cot_text, device)
        non_cot_toks, non_cot_ent = token_entropy(model, tokenizer, non_cot_text, device)
        
        cot_prompt_toks, _ = token_entropy(model, tokenizer, cot_prompt, device)
        non_cot_prompt_toks, _ = token_entropy(model, tokenizer, non_cot_prompt, device)
 

        qa_idx['cot_entropy'] = cot_ent
        qa_idx['cot_token'] = cot_toks
        qa_idx['non_cot_toks'] = non_cot_toks
        qa_idx['non_cot_ent'] = non_cot_ent
        qa_idx['cot_prompt_token_length'] = len(cot_prompt_toks)
        qa_idx['non_cot_prompt_token_length'] = len(non_cot_prompt_toks)
        process_results.append(qa_idx)
    return process_results
    with open(model_name + "_outputs.json", "w", encoding="utf-8") as f:
        json.dump(process_results, f, ensure_ascii=False, indent=2)

# jsonl_path = '/hkfs/home/project/hk-project-p0022560/tum_fmp0582/hallucination/fact_data_reasoning_result_xiaomin/Qwen2.5-1.5B-Instruct/Qwen2.5-1.5B-Instruct-All.jsonl'
# model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
# results = analyze(jsonl_path, model_name,  output_csv=None, use_cuda=True)


jsonl_path = '/hkfs/home/project/hk-project-p0022560/tum_fmp0582/hallucination/fact_data_reasoning_result_xiaomin/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-All.jsonl'
model_name = "meta-llama/Llama-3.2-1B"
results = analyze(jsonl_path, model_name,  output_csv=None, use_cuda=True)




'''
plot image
'''