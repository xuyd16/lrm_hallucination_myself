
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
'''
directly print
'''

def print_colored_sentence(strings, values):
    """
    根据数值为字符串着色并打印成一句话
    :param strings: 字符串列表
    :param values: 对应的数值列表
    """
    # 检查输入有效性
    if len(strings) != len(values):
        raise ValueError("字符串列表和数值列表长度必须相同")
    if not strings:
        return
    
    # 找到最小值和最大值用于归一化
    min_val = min(values)
    max_val = max(values)
    
    # 如果所有值相同，使用中间颜色
    if min_val == max_val:
        normalized_vals = [0.5] * len(values)
    else:
        # 归一化数值到0-1范围
        normalized_vals = [(v - min_val) / (max_val - min_val) for v in values]
    
    # 构建带颜色的字符串列表
    colored_strings = []
    for s, val in zip(strings, normalized_vals):
        # 计算RGB颜色（0-255）
        # 低值 -> 蓝色 (0,0,255), 高值 -> 红色 (255,0,0)
        r = int(255 * val)
        b = int(255 * (1 - val))
        g = 0  # 绿色分量设为0
        
        # 创建ANSI颜色代码
        color_code = f"\033[38;2;{r};{g};{b}m"
        reset_code = "\033[0m"
        
        # 添加颜色到字符串
        colored_strings.append(f"{color_code}{s}{reset_code}")
    
    # 组合所有字符串并打印
    sentence = " ".join(colored_strings)
    print(sentence)
 
# for idx in range(1000):#len(data)):
#     try:
#         cot_tokens = data[idx][0][0] #["Language", " models", " are", " amazing"]
#         cot_entropies = data[idx][0][1] #[1.2, 0.8, 0.5, 2.3]  # 每个token对应的熵值
        
#         response_tokens = data[idx][1][0] #["Language", " models", " are", " amazing"]
#         response_entropies = data[idx][1][1] #[1.2, 0.8, 0.5, 2.3]  # 每个token对应的熵值
        
#         tokens = response_tokens
#         entropies = response_entropies
 
#         if (np.nanmax(entropies) > 0.8):
#             print_colored_sentence(tokens, entropies)
        
#     except:
#         continue

'''
print to html
'''
def save_multiple_colored_html(sequences, filename="colored_text.html"):
    """
    保存多个序列的彩色HTML文件，每个序列独立归一化
    
    参数:
    sequences - 列表，每个元素是一个元组 (tokens, entropies)
    filename - 要保存的文件名
    """
    # 创建HTML内容
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Per-Sequence Entropy Visualization</title>
        <style>
            body { 
                font-family: 'Courier New', monospace; 
                font-size: 16px; 
                line-height: 1.6;
                background-color: #f8f9fa;
                padding: 20px;
            }
            .sequence {
                margin-bottom: 20px;
                padding: 15px;
                border: 1px solid #e0e0e0;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .header {
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
                padding-bottom: 5px;
                border-bottom: 1px solid #eee;
            }
            .legend {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
                font-size: 14px;
                color: #666;
            }
            .color-scale {
                display: flex;
                width: 100%;
                height: 15px;
                margin-bottom: 10px;
                background: linear-gradient(to right, #0000ff, #ff0000);
                border-radius: 3px;
            }
            .scale-labels {
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                color: #666;
            }
            .tokens-container {
                padding: 10px;
                background-color: #fafafa;
                border-radius: 4px;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
    <h1>Per-Sequence Entropy Visualization</h1>
    <p>Each sequence is normalized independently using its own min and max entropy values.</p>
    """
    
    # 添加每个序列
    for i, (tokens, entropies) in enumerate(sequences):
        if not tokens or not entropies:
            continue
            
        # 计算序列的最小、最大和平均熵值
        seq_min = min(entropies)
        seq_max = max(entropies)
        seq_avg = sum(entropies) / len(entropies)
        range_val = seq_max - seq_min
        
        # 序列头
        html_content += f'<div class="sequence">'
        html_content += f'<div class="header">Sequence {i+1} (Min: {seq_min:.3f}, Avg: {seq_avg:.3f}, Max: {seq_max:.3f})</div>'
        
        # 添加颜色刻度和标签
        html_content += f'<div class="legend">'
        html_content += f'<div>Low Entropy (Blue)</div>'
        html_content += f'<div>High Entropy (Red)</div>'
        html_content += f'</div>'
        
        html_content += f'<div class="color-scale"></div>'
        html_content += f'<div class="scale-labels">'
        html_content += f'<span>{seq_min:.3f}</span>'
        html_content += f'<span>{(seq_min + seq_max)/2:.3f}</span>'
        html_content += f'<span>{seq_max:.3f}</span>'
        html_content += f'</div>'
        
        # 添加带颜色的文本
        html_content += '<div class="tokens-container">'
        for token, entropy in zip(tokens, entropies):
            # 归一化熵值 (0-1范围)
            if range_val == 0:
                normalized_entropy = 0.5  # 所有熵值相同的情况
            else:
                normalized_entropy = (entropy - seq_min) / range_val
            
            # 计算RGB颜色
            r = int(255 * normalized_entropy)
            b = int(255 * (1 - normalized_entropy))
            color_hex = f"#{r:02x}00{b:02x}"
            
            # 添加带颜色的span
            html_content += f'<span style="color: {color_hex}; font-weight: {"bold" if normalized_entropy > 0.7 else "normal"};">{token}</span> '
        
        html_content += '</div></div>'  # 结束 tokens-container 和 sequence
    
    # 结束HTML
    html_content += """
    </body>
    </html>
    """
    
    # 保存文件
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"已保存 {len(sequences)} 个序列到 {filename}")
    
# 准备要保存的序列
cot_correctness = []
noncot_correctness = []
cot_sequences_to_save = []
non_cot_sequences_to_save = []
 
for idx in range(1000):  # 或者 len(data)
    try:
        cot_tokens = data[idx]['cot_token']  # 提取tokens
        cot_entropies = data[idx]['cot_entropy']  # 提取熵值
        cot_correctness = data[idx]['cot_response_correct']

        non_cot_tokens = data[idx]['non_cot_toks']
        non_cot_entropies = data[idx]['non_cot_ent']
        non_cot_correctness = data[idx]['response_correct']
      
        noncot_correctness.append(non_cot_correctness)
        cot_correctness.append(cot_correctness)  

        for i in range(2):
            if i == 1:
                tokens, entropies = cot_tokens, cot_entropies
            else: 
                tokens, entropis = non_cot_tokens, non_cot_entropies

            # 检查序列长度和最大熵
            if len(tokens) >= 5 and np.nanmax(entropies) > 0.8:
                if i == 1:
                    cot_sequences_to_save.append((tokens, entropies))
                else:
                    non_cot_sequences_to_save.append((tokens, entropies))
                    
            
            # 可选：在控制台打印
            #print_colored_sentence(tokens, entropies)
            
            # 限制保存的序列数量
            #if len(sequences_to_save) >= 50:
                #break
    
    except Exception as e:
        print(f"处理序列 {idx} 时出错: {str(e)}")
        continue

# 保存所有符合条件的序列到HTML文件
if cot_sequences_to_save:
    save_multiple_colored_html(cot_sequences_to_save, "high_entropy_sequences.html")
else:
    print("没有找到符合条件的序列")