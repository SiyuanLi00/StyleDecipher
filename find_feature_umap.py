import json
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde, entropy 

# 1. 加载特征向量
def load_feature_vectors(file_path):
    """从 JSON 文件加载特征向量。"""
    try:
        with open(file_path, 'r') as f:
            feature_vectors = json.load(f)
        feature_vectors = np.array(feature_vectors, dtype=np.float32)
        feature_vectors[:, 40:56] = feature_vectors[:, 40:56] / 100.0
        return feature_vectors
    except FileNotFoundError:
        print(f"Error: Feature vector file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred loading feature vectors from {file_path}: {e}")
        return None

# 2. 加载包含 Source 信息的条目并生成标签
def load_labels_from_source(file_path):
    """从包含 Source 信息的 JSON 文件加载数据并生成标签。"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        labels = []
        for item in data:
            if item.get("Source") == "human":
                labels.append(0)
            else: # Assuming anything not "human" is "GPT"
                labels.append(1)
        return np.array(labels)
    except FileNotFoundError:
        print(f"Error: Source info file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred loading labels from {file_path}: {e}")
        return None

# 3. 数据标准化
def standardize_data(data):
    """标准化数据（零均值和单位方差）。"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# 4. 使用 UMAP 进行降维
def perform_umap(data, n_components=2, random_state=42):
    """使用 UMAP 进行降维。"""
    # 调整 UMAP 参数以调试结果，例如 n_neighbors, min_dist, metric
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, 
                         n_neighbors=20,
                        min_dist=0.1, 
                         metric='cosine', 
                         n_jobs=-1)
    embedding = reducer.fit_transform(data)
    return embedding

# 5. 可视化降维后的数据并根据标签着色
def visualize_umap_with_labels(principal_components, labels, save_path='umap_labeled.png'):
    """可视化降维后的数据，根据标签着色，并保存图像。"""
    plt.figure(figsize=(10, 8)) # 稍微增大图表大小

    # 定义颜色和标签的映射
    label_map = {0: 'Human', 1: 'GPT'} # 更改标签名称以匹配论文风格
    colors = [('forestgreen' if label == 0 else 'firebrick') for label in labels] # 自定义颜色，更清晰

    # 绘制散点图
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                          c=colors, alpha=1, s=40) # 调整点的大小和透明度

    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.title('2D UMAP Projection of Text Embeddings by Source', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6) # 添加虚线网格


    handles = []
    legend_labels = []
    for label_val in sorted(np.unique(labels)):
        color = 'forestgreen' if label_val == 0 else 'firebrick'
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label_map[label_val],
                                  markerfacecolor=color, markersize=8))
        legend_labels.append(label_map[label_val])
    
    plt.legend(handles=handles, title="Source", loc='best', fontsize=10, title_fontsize='12')

    plt.tight_layout() 
    plt.savefig(save_path, dpi=300) 
    plt.show()
    print(f"Labeled UMAP projection image saved to: {save_path}")


def calculate_distribution_distances(dist1_pdf_values, dist2_pdf_values):
    """
    计算两个概率分布（由其PDF值表示）之间的 Hellinger 距离和 KL 散度。
    输入：两个 numpy 数组，表示在相同评估点上的 PDF 值。
    """

    p = np.maximum(dist1_pdf_values, 0)
    q = np.maximum(dist2_pdf_values, 0)


    epsilon = 1e-10 
    P_norm = (p + epsilon) / np.sum(p + epsilon)
    Q_norm = (q + epsilon) / np.sum(q + epsilon)

    # 计算 KL 散度 (D_KL(P || Q) 和 D_KL(Q || P))
    kl_pq = entropy(P_norm, Q_norm)
    kl_qp = entropy(Q_norm, P_norm)

    # Hellinger Distance 公式
    hellinger_dist = np.sqrt(np.sum((np.sqrt(P_norm) - np.sqrt(Q_norm))**2)) / np.sqrt(2)
    
    return kl_pq, kl_qp, hellinger_dist

if __name__ == "__main__":
    feature_vector_file = 'all_feature_vectors.json'  
    source_info_file = 'all_rewrite_data.json'        
    save_file = 'umap_labeled_projection.png'


    feature_vectors = load_feature_vectors(feature_vector_file)
    labels = load_labels_from_source(source_info_file)

    if feature_vectors is None or labels is None:
        print("Error: Failed to load data. Exiting.")
        exit()

    print("Shape of feature vectors:", feature_vectors.shape)
    print("Shape of labels:", labels.shape)
    print("First few labels:", labels[:10])


    if feature_vectors.shape[0] != labels.shape[0]:
        raise ValueError("The number of feature vectors and labels must be the same.")


    scaled_data, scaler = standardize_data(feature_vectors)


    n_components_to_keep = 2 # 保持 2D 嵌入以方便可视化和 KDE
    umap_embedding = perform_umap(scaled_data, n_components=n_components_to_keep)


    visualize_umap_with_labels(umap_embedding, labels, save_path=save_file)


    print("\nCalculating Distribution Distances (KL Divergence & Hellinger Distance)...")


    human_embeddings = umap_embedding[labels == 0]
    gpt_embeddings = umap_embedding[labels == 1]

    # 检查是否有足够的数据点来拟合 KDE

    if len(human_embeddings) < n_components_to_keep or len(gpt_embeddings) < n_components_to_keep:
        print(f"Not enough data points (need at least {n_components_to_keep}) in one or both categories to perform KDE for distance calculations.")
    else:
        try:

            kde_human = gaussian_kde(human_embeddings.T)
            kde_gpt = gaussian_kde(gpt_embeddings.T)


            x_min, y_min = np.min(umap_embedding, axis=0) - 0.1 
            x_max, y_max = np.max(umap_embedding, axis=0) + 0.1 


            num_grid_points = 200 
            x_grid = np.linspace(x_min, x_max, num_grid_points)
            y_grid = np.linspace(y_min, y_max, num_grid_points)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            eval_points = np.vstack([X_grid.ravel(), Y_grid.ravel()])

            pdf_human_values = kde_human(eval_points)
            pdf_gpt_values = kde_gpt(eval_points)

            kl_pq, kl_qp, hellinger_dist = calculate_distribution_distances(pdf_human_values, pdf_gpt_values)
            
            print(f"KL Divergence (Human || GPT): {kl_pq:.4f}")
            print(f"KL Divergence (GPT || Human): {kl_qp:.4f}")
            print(f"Hellinger Distance (Human, GPT): {hellinger_dist:.4f}")

        except np.linalg.LinAlgError as e:
            print(f"Error during KDE estimation (likely due to singular matrix or insufficient data for covariance): {e}")
            print("This can happen if data points are perfectly collinear or too few for the number of dimensions.")
        except Exception as e:
            print(f"An unexpected error occurred during distance calculations: {e}")

    print("\nProgram finished.")