import numpy as np
import json
from fuzzywuzzy import fuzz
import tiktoken
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, pairwise_distances
from style import get_all_embeddings, create_style_processor, StyleConfig

style_processor, style_model, style_tokenizer, style_params = create_style_processor()

def load_json(filename, default_value):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{filename} 不存在，加载默认值。")
        return default_value
    except Exception as e:
        print(f"加载 {filename} 时出错: {e}")
        return default_value

DOMAIN_PATH = "dataset/Student Essay/essay_mixed_data_combine.json"
REWRITE_DATA_PATH = "mix_result/essay_rewrite_data.json"
FEATURE_VECTORS_PATH = "mix_result/essay_feature_vectors.json"

DOMAIN_PATH = "dataset/att3_combine.json"
REWRITE_DATA_PATH = "RAID_result/att3_rewrite_data.json"
FEATURE_VECTORS_PATH = "RAID_result/att3_feature_vectors.json"

rewrite_data = load_json(REWRITE_DATA_PATH, default_value = [])
feature_vectors = load_json(FEATURE_VECTORS_PATH, default_value = [])



def get_first_1024_tokens(text):
    
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    first_1024_tokens = tokens[:1024]
    decoded_tokens = enc.decode(first_1024_tokens)
    
    return decoded_tokens

def tokenize_and_normalize(sentence):
    return [word.lower().strip() for word in sentence.split()]

def extract_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def common_elements(list1, list2):
    return set(list1) & set(list2)

def calculate_sentence_common(sentence1, sentence2):
    
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)

    common_words = common_elements(tokens1, tokens2)

    number_common_hierarchy = [len(list(common_words))]

    for n in range(2, 5): 
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2) 
        number_common_hierarchy.append(len(list(common_ngrams)))

    return number_common_hierarchy

cutoff_start = 0
cutoff_end = 6000000
ngram_num = 4
def sum_for_list(a,b):
    return [aa+bb for aa, bb in zip(a,b)]


def get_stat(index: int):
    item = rewrite_data[index]
    print(item)
    original = item['Text']
    
    stat = {"Text": original}
    
    raw_embedding = get_all_embeddings(original, style_model, style_tokenizer, style_params)
    # 对 raw_embedding 进行池化（例如取平均值）
    raw_embedding = np.mean(raw_embedding, axis=0)  # 从 (1, seq_len, embedding_dim) -> (embedding_dim,)
    
    raw = tokenize_and_normalize(original)
    
    if len(raw) < cutoff_start or len(raw) > cutoff_end:
        return stat
    else:
        print(item["Index"])
        
    statistic_res = {}
    ratio_fzwz = {}
    style_res = {}
    all_statistic_res = [0 for i in range(ngram_num)]
    cnt = 0
    whole_combined = ''
    
    for pp in item.keys():
        if pp != 'common_features' and pp != 'Index' and pp != "Source":
            whole_combined += (' ' + item[pp])
            
            res = calculate_sentence_common(original, item[pp])
            statistic_res[pp] = res
            all_statistic_res = sum_for_list(all_statistic_res, res)
            
            ratio_fzwz[pp] = [
                fuzz.ratio(original, item[pp]),
                fuzz.token_set_ratio(original, item[pp])
            ]
            
            current_embedding = get_all_embeddings(item[pp], style_model, style_tokenizer, style_params)
            # 对 current_embedding 进行池化
            current_embedding = np.mean(current_embedding, axis=0)  # 从 (1, seq_len, embedding_dim) -> (embedding_dim,)
            
            # 计算余弦距离
            style_res[pp] = pairwise_distances([raw_embedding], [current_embedding], metric="cosine")[0][0]
            
            cnt += 1
       
    print(cnt)     
    stat['fzwz_features'] = ratio_fzwz
    stat['common_features'] = statistic_res
    stat['avg_common_features'] = [a / cnt for a in all_statistic_res]
    stat['common_features_ori_vs_allcombined'] = calculate_sentence_common(original, whole_combined)
    stat['style_features'] = style_res
    
    each_data_fea = [ind_d / len(raw) for ind_d in stat['avg_common_features']]
    
    for ek in stat['common_features'].keys():
        each_data_fea.extend([ind_d / len(raw) for ind_d in stat['common_features'][ek]])

    each_data_fea.extend([ind_d / len(raw) for ind_d in stat['common_features_ori_vs_allcombined']])
    
    for ek in stat['fzwz_features'].keys():
        each_data_fea.extend(stat['fzwz_features'][ek])
    
    for ek in stat['style_features'].keys():
        each_data_fea.append(stat['style_features'][ek])
    
    feature_vectors.append(each_data_fea)


def load_data(address):
    with open(f'{address}', 'r') as f:
        datas = json.load(f)
    return datas        
            
def save_rewrite_data(rewritten_text, prompt, rewrite_item):
    rewrite_item[prompt] = rewritten_text

def xgboost_classifier(data_range : int):
    stack_feature_vectors = np.vstack(feature_vectors)
    
    labels = np.array([0 if _["Source"] == "human" else 1 for _ in rewrite_data[:data_range]])
    
    x_train, x_test, y_train, y_test = train_test_split(stack_feature_vectors, labels, test_size = 0.2, random_state = 42)
    
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    clf = MLPClassifier(
        hidden_layer_sizes=(10,),  # 定义隐藏层大小为 10 个神经元
        max_iter=1000,  # 最大迭代次数为 1000
        activation='relu',  # 激活函数为 ReLU
        solver='adam',  # 优化算法为 Adam
        random_state=42  # 随机种子
    )
    
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    
    # model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=10, random_state=42) # 74.44\%,  turn out for reuter: 62%
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred), "F1 score", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits = 4))
    
def save_rewrite(rewrite_data, filename=REWRITE_DATA_PATH):

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(rewrite_data, f, ensure_ascii=False, indent=4)
        print(f"rewritten texts have been saved to {filename}!")
    except Exception as e:
        print(f"Failed: {e}")

def save_features(feature_vectors, filename=FEATURE_VECTORS_PATH):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(feature_vectors, f, ensure_ascii=False, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        print(f"features we extracted have been saved to {filename}!")
    except Exception as e:
        print(f"Failed: {e}")
    


def save_json(data, filename):
    """将数据保存为 JSON 文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        print(f"数据已保存到 {filename}")
    except Exception as e:
        print(f"保存 {filename} 时出错: {e}")