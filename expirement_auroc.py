import numpy as np
import json
import random
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, confusion_matrix

def load_json(filename, default_value):
    """
    从文件加载 JSON 数据，如果文件不存在或出错则返回默认值。
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: {filename} 未找到，加载默认值。")
        return default_value
    except json.JSONDecodeError:
        print(f"错误: 无法从 {filename} 解码 JSON。返回默认值。")
        return default_value
    except Exception as e:
        print(f"加载 {filename} 时出错: {e}。返回默认值。")
        return default_value

def prepare_combined_data_by_order(rewrite_data, feature_vectors):
    """
    根据列表顺序组合重写数据和特征向量。
    返回一个字典列表，每个字典包含“Vectors”和“Label”。
    要求 rewrite_data 和 feature_vectors 长度相同且顺序对应。
    """
    if len(rewrite_data) != len(feature_vectors):
        print("错误: rewrite_data 和 feature_vectors 的长度不匹配，无法按顺序组合。")
        return []

    combined_data = []
    skipped_count = 0
    for i in range(len(rewrite_data)):
        item = rewrite_data[i]
        vector = feature_vectors[i] # 直接按顺序获取对应的向量

        source = item.get("Source")

        if source is None:
            print(f"警告: 跳过索引 {i} 的项目，因缺少“Source”字段: {item}")
            skipped_count += 1
            continue
        
        # 确保“human”标签为 0，“GPT”标签为 1
        label = 0 if source == "human" else 1 # 假设“GPT”表示机器生成

        combined_data.append({
            "Label": label,
            "Vectors": np.array(vector) # 将列表转换为 numpy 数组以保持一致性
        })
    
    if skipped_count > 0:
        print(f"因数据缺失或格式问题而跳过 {skipped_count} 个项目。")
    return combined_data

def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, scaler, classifier_type="xgboost"):
    """
    训练和评估指定分类器（XGBoost 或 MLPClassifier）。
    打印准确率、分类报告、F1 分数和 AUROC。
    """
    print(f"\n--- 训练和评估 {classifier_type.upper()} ---")

    model = None
    if classifier_type == "xgboost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        X_train_scaled = X_train
        X_test_scaled = X_test
    elif classifier_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        print("错误: 无效的 classifier_type。请选择 'xgboost' 或 'mlp'。")
        return

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)

    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"F1 分数 (F1-score): {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("\n分类报告 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=["human", "GPT"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵 (Confusion Matrix):")
    print(f"真阴性 (True Negatives, human 被正确分类为 human): {cm[0, 0]}")
    print(f"假阳性 (False Positives, human 被错误分类为 GPT): {cm[0, 1]}")
    print(f"假阴性 (False Negatives, GPT 被错误分类为 human): {cm[1, 0]}")
    print(f"真阳性 (True Positives, GPT 被正确分类为 GPT): {cm[1, 1]}")


# --- 数据加载 ---
print("正在加载数据...")

# 将数据集信息存储在一个字典中，方便迭代
datasets_info = {
    "Essay": {
        "rewrite_data": load_json("Essay_result/rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("Essay_result/feature_vectors.json", default_value=[])
    },
    "Abstract": {
        "rewrite_data": load_json("Abstract_result/rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("Abstract_result/feature_vectors.json", default_value=[])
    },
    "Code": {
        "rewrite_data": load_json("Code_result/rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("Code_result/feature_vectors.json", default_value=[])
    },
    "News": {
        "rewrite_data": load_json("News_result/rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("News_result/feature_vectors.json", default_value=[])
    },
    "Yelp": {
        "rewrite_data": load_json("Yelp_result/rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("Yelp_result/feature_vectors.json", default_value=[])
    },
    "PEssay": {
        "rewrite_data": load_json("perturbed_result/essay_rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("perturbed_result/essay_feature_vectors.json", default_value=[])
    },
    "PCode": {
        "rewrite_data": load_json("perturbed_result/code_rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("perturbed_result/code_feature_vectors.json", default_value=[])
    },
    "PNews": {
        "rewrite_data": load_json("perturbed_result/news_rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("perturbed_result/news_feature_vectors.json", default_value=[])
    },
    "PYelp": {
        "rewrite_data": load_json("perturbed_result/yelp_rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("perturbed_result/yelp_feature_vectors.json", default_value=[])
    },
}

print("数据加载完成。")

# --- 循环遍历每个数据集并进行分类 ---
scaler = StandardScaler() # StandardScaler 在每次数据集循环时会被 fit_transform 重新初始化

for dataset_name, data_pair in datasets_info.items():
    print(f"\n{'='*50}")
    print(f"--- 正在处理数据集: {dataset_name} ---")
    print(f"{'='*50}")

    rewrite_data = data_pair["rewrite_data"]
    feature_vectors = data_pair["feature_vectors"]

    print(f"数据集 '{dataset_name}' 的重写数据项: {len(rewrite_data)}")
    print(f"数据集 '{dataset_name}' 的特征向量数: {len(feature_vectors)}")

    # 为当前数据集准备数据
    combined_dataset = prepare_combined_data_by_order(rewrite_data, feature_vectors)

    if not combined_dataset:
        print(f"数据集 '{dataset_name}' 没有有效的分类数据。跳过。")
        continue
    else:
        print(f"成功准备了 {len(combined_dataset)} 个 '{dataset_name}' 数据点。")

    # 分离特征 (X) 和标签 (y)
    X = np.array([item["Vectors"][:56] for item in combined_dataset])
    y = np.array([item["Label"] for item in combined_dataset])

    print(f"数据集 '{dataset_name}' 的 X 形状: {X.shape}")
    print(f"数据集 '{dataset_name}' 的 y 形状: {y.shape}")

    # 检查特征向量维度是否一致
    if X.ndim != 2 or X.shape[0] != len(combined_dataset):
        print(f"错误: 数据集 '{dataset_name}' 的特征向量形状不一致或组合不正确。跳过。")
        continue
    
    # 检查是否存在两个类别，以便计算 AUROC
    if len(np.unique(y)) < 2:
        print(f"错误: 数据集 '{dataset_name}' 中只存在一个类别。无法计算 AUROC 或执行有意义的二分类。跳过。")
        continue

    # 将数据分割为训练集和测试集
    # 对于每个数据集，都独立进行训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"数据集 '{dataset_name}' 训练集大小: {len(X_train)}")
    print(f"数据集 '{dataset_name}' 测试集大小: {len(X_test)}")

    # --- 训练和评估 XGBoost ---
    train_and_evaluate_classifier(X_train, X_test, y_train, y_test, scaler, classifier_type="xgboost")

    # --- 训练和评估 MLPClassifier ---
    train_and_evaluate_classifier(X_train, X_test, y_train, y_test, scaler, classifier_type="mlp")

print("\n所有数据集的分类过程完成。")