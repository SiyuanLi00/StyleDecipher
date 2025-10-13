import numpy as np
import json
from fuzzywuzzy import fuzz
import tiktoken
import random
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

def load_json(filename, default_value):
    """从文件加载 JSON 数据，如果文件不存在则返回默认值"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{filename} 不存在，加载默认值。")
        return default_value
    except Exception as e:
        print(f"加载 {filename} 时出错: {e}")
        return default_value

def get_classify_data(data, vectors):
    stack_feature_vectors = np.vstack(vectors)
    labels = np.array([0 if _["Source"] == "human" else 1 for _ in data])
    classify_data = []
    for i in range(len(labels)):
        classify_data.append({
            "Label": labels[i],
            "Vectors": stack_feature_vectors[i]
        })
    return classify_data



E_rewrite_data = load_json("Essay_result/rewrite_data.json", default_value = [])
E_feature_vectors = load_json("Essay_result/feature_vectors.json", default_value = [])

A_rewrite_data = load_json("Abstract_result/rewrite_data.json", default_value = [])
A_feature_vectors = load_json("Abstract_result/feature_vectors.json", default_value = [])

C_rewrite_data = load_json("Code_result/rewrite_data.json", default_value = [])
C_feature_vectors = load_json("Code_result/feature_vectors.json", default_value = [])

N_rewrite_data = load_json("News_result/rewrite_data.json", default_value = [])
N_feature_vectors = load_json("News_result/feature_vectors.json", default_value = [])

Y_rewrite_data = load_json("Yelp_result/rewrite_data.json", default_value = [])
Y_feature_vectors = load_json("Yelp_result/feature_vectors.json", default_value = [])



def xgboost_classifier(train_data, train_vectors, test_data, test_vectors):
    
    train_dataset = get_classify_data(train_data, train_vectors)
    test_dataset = get_classify_data(test_data, test_vectors)
    
    scale = int(len(train_dataset) * 0.2)
    test_dataset = random.sample(test_dataset, scale)

    x_train = [_["Vectors"][:44] for _ in train_dataset]
    x_test = [np.concatenate((_["Vectors"][:32], _["Vectors"][40:52])) for _ in test_dataset]
    y_train = [_["Label"] for _ in train_dataset]
    y_test = [_["Label"] for _ in test_dataset]
    
    
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
    
    model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=10, random_state=42) # 74.44\%,  turn out for reuter: 62%
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred), "F1 score", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits = 4))
    
    
if __name__ == "__main__":
    train_data = C_rewrite_data
    train_vectors = C_feature_vectors
    test_data = Y_rewrite_data
    test_vectors = Y_feature_vectors

    xgboost_classifier(train_data, train_vectors, test_data, test_vectors)