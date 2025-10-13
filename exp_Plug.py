import numpy as np
import json
from fuzzywuzzy import fuzz
from sklearn.metrics import accuracy_score, classification_report, f1_score, pairwise_distances
from transformers import AutoModel, AutoTokenizer
import torch 
from style import get_all_embeddings, create_style_processor, StyleConfig

style_processor, style_model, style_tokenizer, style_params = create_style_processor() 

def tokenize_and_normalize(sentence):
    return [word.lower().strip() for word in sentence.split()]

def extract_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def common_elements(list1, list2):
    return set(list1) & set(list2)

def calculate_sentence_common(sentence1, sentence2):
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)

    number_common_hierarchy = [len(common_elements(tokens1, tokens2))] 

    for n in range(2, 5): 
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2) 
        number_common_hierarchy.append(len(list(common_ngrams)))
    
    return number_common_hierarchy

def sum_for_list(a, b):
    return [aa + bb for aa, bb in zip(a, b)]
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_embedding(text: str, vectorizer_obj: TfidfVectorizer) -> np.ndarray:
    """
    使用预训练的 TfidfVectorizer 生成文本的 TF-IDF 嵌入。
    """
    tfidf_vector = vectorizer_obj.transform([text])
    return tfidf_vector.toarray().flatten()


# 方法 B: Word2Vec/GloVe 平均池化嵌入函数
import gensim.downloader as api # 可以替换为 gensim.models.KeyedVectors 加载本地文件

def get_word_avg_embedding(text: str, word_vectors_model) -> np.ndarray:
    """
    使用预训练词向量模型（如 GloVe/Word2Vec）对句子进行平均池化。
    """
    tokens = tokenize_and_normalize(text)
    valid_vectors = [word_vectors_model[word] for word in tokens if word in word_vectors_model]
    
    if len(valid_vectors) == 0:
        # 如果没有一个词在词汇表中，返回一个零向量
        # 确保这个零向量的维度与模型向量维度一致
        if hasattr(word_vectors_model, 'vector_size'):
             return np.zeros(word_vectors_model.vector_size)
        elif hasattr(word_vectors_model, 'vector_size'): # Fallback for models without direct 'vector_size' if needed
             # This assumes all vectors in the model have the same size.
             # You might need to pick a random existing vector's size.
             # Or, if you know the dim, hardcode it.
             return np.zeros(list(word_vectors_model.values())[0].shape[0]) if len(word_vectors_model) > 0 else np.zeros(100) # Default to 100
        else: # Fallback if no vector_size attr or no vectors
             print("警告：无法确定词向量维度，返回一个默认大小的零向量（维度100）。")
             return np.zeros(100) # 假设一个默认维度
    else:
        return np.mean(valid_vectors, axis=0)


# 方法 C: BERT/RoBERTa 嵌入函数
def get_bert_roberta_embedding(text: str, model_obj, tokenizer_obj, pool_type: str = 'cls') -> np.ndarray:
    """
    使用 BERT/RoBERTa 模型生成文本嵌入。
    Args:
        text (str): 输入文本。
        model_obj: 已加载的 BERT/RoBERTa 模型实例。
        tokenizer_obj: 已加载的 BERT/RoBERTa 分词器实例。
        pool_type (str): 池化类型，'cls' 或 'mean'。
    Returns:
        np.ndarray: 文档向量（1D）。
    """
    inputs = tokenizer_obj(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    device = model_obj.device if hasattr(model_obj, 'device') else 'cpu'
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_obj(**inputs)
    
    if pool_type == 'cls':
        pooled_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    elif pool_type == 'mean':
        pooled_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
    else:
        raise ValueError("pool_type 必须是 'cls' 或 'mean'")
    
    if pooled_embedding.ndim > 1:
        pooled_embedding = pooled_embedding.squeeze()
        
    return pooled_embedding


# --- 通用特征提取函数（调整以接收通用嵌入函数及其参数） ---
def get_feature_vector(
    item: dict,
    embedding_func: callable, # 接收一个可调用的嵌入函数
    embedding_params: dict, # 接收该嵌入函数所需的额外参数
    cutoff_start: int,
    cutoff_end: int,
    ngram_num: int, # 这个参数现在表示 calculate_sentence_common 将返回的 n-gram 数量
) -> list | None:
    """
    计算单个数据集 item 的特征向量。

    参数:
        item (dict): 数据集中的一个条目，预期包含 'Text' 和可能的 'RewriteX' 字段。
        embedding_func (callable): 用于生成嵌入的函数（如 get_tfidf_embedding, get_word_avg_embedding 等）。
        embedding_params (dict): 传递给 embedding_func 的参数字典。
        cutoff_start (int): 原始文本被处理的最小 token 长度阈值。
        cutoff_end (int): 原始文本被处理的最大 token 长度阈值。
        ngram_num (int): calculate_sentence_common 函数将返回的 n-gram 特征的数量（例如，4 代表 1-gram 到 4-gram）。

    返回:
        list | None: 如果 item 的原始文本符合长度标准，则返回包含所有特征的数值列表；
                     否则返回 None。
    """
    original_text = item.get('Text')
    if original_text is None:
        return None

    raw_tokens = tokenize_and_normalize(original_text)

    if len(raw_tokens) < cutoff_start or len(raw_tokens) > cutoff_end:
        return None
    
    each_data_fea = []

    # 调用传入的嵌入函数
    raw_embedding = embedding_func(original_text, **embedding_params)

    style_features_list = [] 
    
    # 根据 calculate_sentence_common 的行为，其返回长度固定为 4（1到4-gram）
    avg_common_features_sum = [0 for _ in range(ngram_num)] 
    rewritten_common_features_list = [] 
    fzwz_features_list = [] 

    whole_combined_text = original_text 
    rewritten_count = 0

    metadata_keys = {'Text', 'common_features', 'Index', 'Source'}
    
    for key in item.keys():
        if key not in metadata_keys:
            rewritten_text = item.get(key)
            if rewritten_text is None:
                continue 
            
            whole_combined_text += (' ' + rewritten_text)
            rewritten_count += 1

            # 1. 计算 N-gram 共同特征
            res_common = calculate_sentence_common(original_text, rewritten_text) 
            rewritten_common_features_list.extend([c / len(raw_tokens) for c in res_common]) 
            
            avg_common_features_sum = sum_for_list(avg_common_features_sum, res_common)

            # 2. 计算模糊比率
            fzwz_features_list.extend([
                fuzz.ratio(original_text, rewritten_text),
                fuzz.token_set_ratio(original_text, rewritten_text)
            ])
            
            # 3. 计算风格特征（嵌入的余弦距离）
            current_embedding = embedding_func(rewritten_text, **embedding_params)
            
            cosine_dist = pairwise_distances([raw_embedding], [current_embedding], metric="cosine")[0][0]
            style_features_list.append(cosine_dist)

    # --- 合并所有特征 ---

    # 1. 平均共同特征
    if rewritten_count > 0:
        each_data_fea.extend([a / rewritten_count / len(raw_tokens) for a in avg_common_features_sum])
    else:
        each_data_fea.extend([0.0] * ngram_num) 

    # 2. 每个改写文本的共同特征
    each_data_fea.extend(rewritten_common_features_list)

    # 3. 原始文本与所有组合改写文本的共同特征
    common_ori_vs_allcombined = calculate_sentence_common(original_text, whole_combined_text)
    each_data_fea.extend([c / len(raw_tokens) for c in common_ori_vs_allcombined])
    
    # 4. 每个改写文本的模糊特征
    each_data_fea.extend(fzwz_features_list)
    
    # 5. 每个改写文本的风格特征
    each_data_fea.extend(style_features_list)
    
    return each_data_fea



def main(input_file: str, output_features_file_prefix: str, output_labels_file: str):
    """
    主函数：加载数据，提取特征，生成标签，并保存结果。
    它将循环运行四种不同的嵌入方法。

    参数:
        input_file (str): 包含原始数据（JSON 格式）的输入文件路径。
        output_features_file_prefix (str): 用于保存提取出的特征向量的文件名前缀。
                                            实际文件名会是 {prefix}_{method_name}.json。
        output_labels_file (str): 用于保存对应标签（JSON 格式）的输出文件路径。
    """
    
    # --- 配置参数 ---
    NGRAM_NUM = 4       # calculate_sentence_common 返回的 n-gram 数量（1-gram 到 4-gram）
    CUTOFF_START = 5    # 原始文本的最小 token 长度
    CUTOFF_END = 20000    # 原始文本的最大 token 长度

    print(f"正在从 {input_file} 加载数据...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            rewrite_data = json.load(f)
        print(f"成功加载 {len(rewrite_data)} 个数据条目。")
    except FileNotFoundError:
        print(f"错误：未找到输入文件 {input_file}。请检查路径。")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解码 JSON 文件 {input_file}。请检查文件格式。")
        return
    except Exception as e:
        print(f"加载数据时发生意外错误：{e}")
        return

    # --- 预加载和预训练所有需要的模型/向量化器 ---

    # 1. TF-IDF Vectorizer (方法 A)
    print("\n[TF-IDF] 正在收集所有文本以训练 TfidfVectorizer...")
    all_texts_for_vectorizer = []
    for item in rewrite_data:
        if 'Text' in item and item['Text'] is not None:
            all_texts_for_vectorizer.append(item['Text'])
        for key in item.keys():
            if key not in {'Text', 'common_features', 'Index', 'Source'} and item[key] is not None:
                all_texts_for_vectorizer.append(item[key])
    
    global_tfidf_vectorizer = TfidfVectorizer(max_features=5000) # 限制词汇表大小
    global_tfidf_vectorizer.fit(all_texts_for_vectorizer)
    print(f"[TF-IDF] TfidfVectorizer 训练完成。词汇表大小: {len(global_tfidf_vectorizer.vocabulary_)}")

    # 2. GloVe/Word2Vec 模型 (方法 B)
    print("\n[Word2Vec/GloVe] 正在加载预训练词向量模型 (glove-wiki-gigaword-100)...")
    global_glove_model = None
    try:
        global_glove_model = api.load("glove-wiki-gigaword-100") 
        print(f"[Word2Vec/GloVe] GloVe 模型加载完成。向量维度: {global_glove_model.vector_size}")
    except Exception as e:
        print(f"错误：加载 GloVe 模型失败。请检查网络或路径。将跳过此方法。错误: {e}")
        # global_glove_model 保持为 None

    # 3. BERT/RoBERTa 模型 (方法 C)
    print("\n[BERT/RoBERTa] 正在加载预训练 BERT 模型 (bert-base-uncased)...")
    global_bert_model = None
    global_bert_tokenizer = None
    bert_model_name = "bert-base-uncased" # 你可以选择 'roberta-base' 等
    try:
        global_bert_model = AutoModel.from_pretrained(bert_model_name)
        global_bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        print(f"[BERT/RoBERTa] 模型 '{bert_model_name}' 加载完成。")
        # 将模型移动到 GPU 如果可用
        if torch.cuda.is_available():
            global_bert_model.to('cuda')
            print("模型已移动到 GPU。")
        else:
            print("GPU 不可用，模型在 CPU 上运行。")
    except Exception as e:
        print(f"错误：加载 BERT 模型 '{bert_model_name}' 失败。请检查模型名称或网络。将跳过此方法。错误: {e}")
        # global_bert_model 和 global_bert_tokenizer 保持为 None

    # 4. SBERT/BGE 模型 (方法 D) - 已从 style 模块导入，假设它们是预加载的
    print("\n[SBERT/BGE] 将使用从 'style' 模块导入的模型和分词器。")
    # 如果 style.model 或 style.tokenizer 为 None，也应该在此处进行检查并跳过
    if model is None or tokenizer is None:
        print("警告：style 模块中的 SBERT/BGE 模型或分词器未正确加载。SBERT/BGE 方法将跳过。")
        sbert_bge_available = False
    else:
        sbert_bge_available = True
    sbert_bge_available = False
    # 定义要测试的嵌入方法集合
    # 结构: "方法名称": (嵌入函数, 嵌入函数所需的参数字典)
    experiment_setups = {}

    if sbert_bge_available:
        experiment_setups["sbert_bge"] = (get_all_embeddings, {'model_param': style_model, 'tokenizer_param': style_tokenizer, 'params_param': style_params})
    experiment_setups["tfidf"] = (get_tfidf_embedding, {'vectorizer_obj': global_tfidf_vectorizer})
    if global_glove_model:
        experiment_setups["glove_avg"] = (get_word_avg_embedding, {'word_vectors_model': global_glove_model})
    if global_bert_model and global_bert_tokenizer:
        experiment_setups["bert_cls"] = (get_bert_roberta_embedding, {'model_obj': global_bert_model, 'tokenizer_obj': global_bert_tokenizer, 'pool_type': 'cls'})
        experiment_setups["bert_mean"] = (get_bert_roberta_embedding, {'model_obj': global_bert_model, 'tokenizer_obj': global_bert_tokenizer, 'pool_type': 'mean'})

    if not experiment_setups:
        print("没有可用的嵌入方法。请检查模型加载情况。")
        return

    # 循环进行不同方法的实验
    # 标签文件只需要保存一次，因为它们与特征提取方法无关
    # 在这个循环中，all_labels 会是最后一次循环生成的，所有标签应该都一样。
    # 为了避免重复生成，可以在循环外单独生成并保存标签。
    first_run = True # 标记是否是第一次循环，用于保存标签
    
    for method_name, (embedding_func, embedding_params) in experiment_setups.items():
        print(f"\n--- 开始 {method_name} 方法的特征提取 ---")
        current_feature_vectors = []
        current_labels = [] # 每次循环生成一份标签，确保与 current_feature_vectors 对应

        processed_count = 0
        skipped_count = 0

        for i, item in enumerate(rewrite_data):
            if (i + 1) % 100 == 0:
                print(f"  [{method_name}] 已处理 {i + 1}/{len(rewrite_data)} 个条目...")

            feature_vector = get_feature_vector(
                item=item,
                embedding_func=embedding_func,
                embedding_params=embedding_params,
                cutoff_start=CUTOFF_START,
                cutoff_end=CUTOFF_END,
                ngram_num=NGRAM_NUM
            )

            if feature_vector is not None:
                current_feature_vectors.append(feature_vector)
                # 标签生成逻辑（与方法无关）
                if item.get("Source") == "human":
                    current_labels.append(0) # Human 标签为 0
                else:
                    current_labels.append(1) # 其他（如 GPT）标签为 1
                processed_count += 1
            else:
                skipped_count += 1
        
        print(f"\n[{method_name}] 特征提取完成。")
        print(f"  成功处理条目数量: {processed_count}")
        print(f"  跳过条目数量: {skipped_count}")
        print(f"  总 {method_name} 特征向量数量: {len(current_feature_vectors)}")

        if not current_feature_vectors:
            print(f"[{method_name}] 没有提取到任何特征向量。跳过保存。")
            continue

        # 保存特征向量（为不同方法使用不同的文件名）
        current_output_features_file = f"{output_features_file_prefix}_{method_name}.json"
        print(f"正在保存 {method_name} 的特征向量到 {current_output_features_file}...")
        try:
            # 将 NumPy 数组转换为列表以便 JSON 序列化
            serializable_feature_vectors = [vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in current_feature_vectors]
            with open(current_output_features_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_feature_vectors, f, ensure_ascii=False, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
            print(f"特征向量已保存到 {current_output_features_file}")
        except Exception as e:
            print(f"保存 {method_name} 特征向量时发生错误：{e}")

        # 标签文件只需要保存一次，因为它们对于所有方法都是一样的
        if first_run:
            print(f"\n正在保存标签到 {output_labels_file}...")
            try:
                with open(output_labels_file, 'w', encoding='utf-8') as f:
                    json.dump(current_labels, f, ensure_ascii=False, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
                print(f"标签已保存到 {output_labels_file}")
                first_run = False # 标记为已保存
            except Exception as e:
                print(f"保存标签时发生错误：{e}")

# --- 程序入口 ---
if __name__ == "__main__":
    # 定义输入和输出文件路径
    INPUT_DATA_FILE = 'Yelp_result/rewrite_data.json' # 原始数据集文件
    OUTPUT_FEATURES_FILE_PREFIX = 'Yelp_result/all_feature_vectors' # 提取特征后保存的文件前缀
    OUTPUT_LABELS_FILE = 'Yelp_result/all_labels.json' # 对应标签保存的文件

    main(INPUT_DATA_FILE, OUTPUT_FEATURES_FILE_PREFIX, OUTPUT_LABELS_FILE)