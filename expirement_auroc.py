import numpy as np
import json
import random
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import entropy

def load_json(filename, default_value):
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
    if len(rewrite_data) != len(feature_vectors):
        print("错误: rewrite_data 和 feature_vectors 的长度不匹配，无法按顺序组合。")
        return []

    combined_data = []
    skipped_count = 0
    for i in range(len(rewrite_data)):
        item = rewrite_data[i]
        vector = feature_vectors[i]

        source = item.get("Source")

        if source is None:
            print(f"警告: 跳过索引 {i} 的项目，因缺少'Source'字段: {item}")
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

def calculate_kl_hellinger(y_true, y_pred_proba, num_bins=20):
    """
    计算 Human 和 GPT 预测概率分布之间的 KL 散度和 Hellinger 距离。
    y_pred_proba 是模型预测为正类 (GPT) 的概率。
    """
    # 按照真实标签分离 Human 和 GPT 的预测概率
    human_probas = y_pred_proba[y_true == 0]
    gpt_probas = y_pred_proba[y_true == 1]

    if len(human_probas) == 0 or len(gpt_probas) == 0:
        print("警告: 无法计算 KL 散度和 Hellinger 距离，因为某个类别的样本数量为零。")
        return None, None, None, None

    # 对概率进行分箱，生成概率分布（PMF的近似）
    # 范围为 [0, 1]，bins 数量
    bins = np.linspace(0, 1, num_bins + 1)
    
    hist_human, _ = np.histogram(human_probas, bins=bins, density=True) # density=True 归一化为概率密度
    hist_gpt, _ = np.histogram(gpt_probas, bins=bins, density=True)
    
    # 避免 log(0) 问题，将 0 值替换为非常小的正数
    epsilon = 1e-10
    P = hist_human + epsilon
    Q = hist_gpt + epsilon
    
    # 归一化以确保它们是有效的概率分布 (和为1)
    P = P / P.sum()
    Q = Q / Q.sum()

    # 计算 KL 散度
    # D_KL(P || Q)
    kl_pq = entropy(P, Q)
    # D_KL(Q || P)
    kl_qp = entropy(Q, P)

    # 计算 Hellinger 距离
    # H(P, Q) = 1/sqrt(2) * ||sqrt(P) - sqrt(Q)||_2
    hellinger_dist = np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2)) / np.sqrt(2)

    return kl_pq, kl_qp, hellinger_dist, num_bins


def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, scaler, classifier_type="xgboost"):
    """
    训练和评估指定分类器（XGBoost 或 MLPClassifier）。
    打印准确率、分类报告、F1 分数和 AUROC。
    添加了计算 Human 和 GPT 预测概率分布的 KL 散度和 Hellinger 距离。
    """
    print(f"\n--- 训练和评估 {classifier_type.upper()} ---")

    model = None
    if classifier_type == "xgboost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        X_train_scaled = X_train
        X_test_scaled = X_test
    elif classifier_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
        # scaler 应该在训练数据上 fit，然后用于转换训练和测试数据
        X_train_scaled = scaler.fit_transform(X_train) # 每次调用都会重新 fit，这对于单次数据集处理是正确的
        X_test_scaled = scaler.transform(X_test)
    else:
        print("错误: 无效的 classifier_type。请选择 'xgboost' 或 'mlp'。")
        return

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # 获取预测为正类 (GPT) 的概率
    # predict_proba 返回 [概率_类别0, 概率_类别1]，我们取类别1的概率
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

    # --- 添加 KL 散度和 Hellinger 距离计算 ---
    kl_pq, kl_qp, hellinger_dist, num_bins_used = calculate_kl_hellinger(y_test, y_pred_proba, num_bins=20)
    
    if kl_pq is not None:
        print(f"\n--- 预测概率分布分析 (基于 {num_bins_used} 个分箱) ---")
        print(f"KL 散度 (Human || GPT): {kl_pq:.4f}") # 衡量从 Human 分布到 GPT 分布的“信息增益”
        print(f"KL 散度 (GPT || Human): {kl_qp:.4f}") # 衡量从 GPT 分布到 Human 分布的“信息增益”
        print(f"Hellinger 距离 (Human, GPT): {hellinger_dist:.4f}")


# --- 数据加载 ---
print("正在加载数据...")

# 将数据集信息存储在一个字典中，方便迭代
datasets_info = {
    "Essay": {
        "rewrite_data": load_json("Essay_result/rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("Essay_result/feature_vectors.json", default_value=[])
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
    "att1": {
        "rewrite_data": load_json("RAID_result/att1_rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("RAID_result/att1_feature_vectors.json", default_value=[])
    },
    "att2": {
        "rewrite_data": load_json("RAID_result/att2_rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("RAID_result/att2_feature_vectors.json", default_value=[])
    },
    "att3": {
        "rewrite_data": load_json("RAID_result/att3_rewrite_data.json", default_value=[]),
        "feature_vectors": load_json("RAID_result/att3_feature_vectors.json", default_value=[])
    },
}

print("数据加载完成。")

# --- 循环遍历每个数据集并进行分类 ---

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
    # 注意：这里你将特征向量截断到前56维，请确保这是你的意图，并且所有特征向量至少有56维。
    X = np.array([item["Vectors"] for item in combined_dataset])
    y = np.array([item["Label"] for item in combined_dataset])

    print(f"数据集 '{dataset_name}' 的 X 形状: {X.shape}")
    print(f"数据集 '{dataset_name}' 的 y 形状: {y.shape}")

    # 检查特征向量维度是否一致
    if X.ndim != 2 or X.shape[0] != len(combined_dataset) or X.shape[1] == 0:
        print(f"错误: 数据集 '{dataset_name}' 的特征向量形状不一致、为空或组合不正确。跳过。")
        continue
    
    # 检查是否存在两个类别，以便计算 AUROC 和 KL/Hellinger
    if len(np.unique(y)) < 2:
        print(f"错误: 数据集 '{dataset_name}' 中只存在一个类别。无法计算 AUROC、KL 散度、Hellinger 距离或执行有意义的二分类。跳过。")
        continue

    # 将数据分割为训练集和测试集
    # 对于每个数据集，都独立进行训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"数据集 '{dataset_name}' 训练集大小: {len(X_train)}")
    print(f"数据集 '{dataset_name}' 测试集大小: {len(X_test)}")

    # 每次迭代为 MLPClassifier 重新实例化 StandardScaler
    # 因为不同的数据集可能有不同的统计特性
    current_scaler = StandardScaler() 

    # --- 训练和评估 XGBoost ---
    train_and_evaluate_classifier(X_train, X_test, y_train, y_test, current_scaler, classifier_type="xgboost")

    # --- 训练和评估 MLPClassifier ---
    train_and_evaluate_classifier(X_train, X_test, y_train, y_test, current_scaler, classifier_type="mlp")

print("\n所有数据集的分类过程完成。")