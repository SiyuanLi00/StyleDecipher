import json
import random
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def get_sentences_in_paragraph(paragraph):
    """
    将一个段落分解成句子并返回这些句子的列表。
    """
    sentences = nltk.sent_tokenize(paragraph)
    return sentences

def read_data(json_path):
    """
    读取指定路径的JSON文件并返回其内容。
    """
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_custom_mixed_data(json_path, output_json_path):
    data = read_data(json_path)
    
    new_mixed_data = []
    
    # 所有的 prompt 键
    prompt_keys = ['Revise the code with your best effort', 'Help me polish this code', 'Rewrite the code with GPT style', 'Refine the code for me please', 'Concise the code without change the functionality']

    
    random.seed(2023) # 为了结果的可复现性

    for item in data:
        original_text = item.get("Text", "")
        
        # 1. 从 'Text' 字段中随机选择一句
        text_sentences = get_sentences_in_paragraph(original_text)
        if not text_sentences:
            continue # 如果Text为空，则跳过此条目
        selected_text_sentence = random.choice(text_sentences)
        
        # 2. 从 prompt_keys 中随机选择 4 个不同的 prompt
        if len(prompt_keys) < 4:
            print(f"Warning: Not enough prompt keys to choose 4 from for index {item.get('Index')}. Skipping.")
            continue
            
        selected_prompts = random.sample(prompt_keys, 4)
        
        selected_prompt_sentences = []
        for prompt_key in selected_prompts:
            prompt_text = item.get(prompt_key, "")
            prompt_sentences = get_sentences_in_paragraph(prompt_text)
            if prompt_sentences:
                selected_prompt_sentences.append(random.choice(prompt_sentences))
        
        # 3. 组合成新的文本
        # 确保我们有足够的句子来混合
        if len(selected_prompt_sentences) == 4:
            final_sentences = [selected_text_sentence] + selected_prompt_sentences
            random.shuffle(final_sentences) # 随机打乱顺序
            new_text = " ".join(final_sentences)
            
            # 4. 组合新的条目
            new_item = {
                "Index": item.get("Index"),
                "Text": new_text,
                "Source": "GPT" # Source 固定为 "GPT"
            }
            new_mixed_data.append(new_item)
        else:
            print(f"Warning: Could not get 4 sentences from selected prompts for index {item.get('Index')}. Skipping.")

    # 5. 保存为新的JSON文件
    with open(output_json_path, 'w', encoding="utf-8") as f:
        json.dump(new_mixed_data[:200], f, ensure_ascii=False, indent=4)
    
    print(f"Successfully created mixed data and saved to {output_json_path}. Total items: {len(new_mixed_data)}")
    return new_mixed_data


create_custom_mixed_data('Code_result/rewrite_data.json', 'dataset/HumanEval Code/code_mixed_data_GPT.json')