import json
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde, entropy 

# 1. Load feature vectors
def load_feature_vectors(file_path):
    """Load feature vectors from JSON file."""
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

# 2. Load entries containing Source information and generate labels
def load_labels_from_source(file_path):
    """Load data from JSON file containing Source information and generate labels."""
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

# 3. Data standardization
def standardize_data(data):
    """Standardize data (zero mean and unit variance)."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# 4. Perform dimensionality reduction using UMAP
def perform_umap(data, n_components=2, random_state=42):
    """Perform dimensionality reduction using UMAP."""
    # Adjust UMAP parameters to debug results, e.g., n_neighbors, min_dist, metric
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, 
                         n_neighbors=20,
                        min_dist=0.1, 
                         metric='cosine', 
                         n_jobs=-1)
    embedding = reducer.fit_transform(data)
    return embedding

# 5. Visualize dimensionality-reduced data colored by labels
def visualize_umap_with_labels(principal_components, labels, save_path='umap_labeled.png'):
    """Visualize dimensionality-reduced data, colored by labels, and save the image."""
    plt.figure(figsize=(10, 8)) # Slightly increase chart size

    # Define color and label mapping
    label_map = {0: 'Human', 1: 'GPT'} # Change label names to match paper style
    colors = [('forestgreen' if label == 0 else 'firebrick') for label in labels] # Custom colors, clearer

    # Draw scatter plot
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                          c=colors, alpha=1, s=40) # Adjust point size and transparency

    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.title('2D UMAP Projection of Text Embeddings by Source', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6) # Add dashed grid


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

# 6. Calculate distribution distances (KL divergence and Hellinger distance)
def calculate_distribution_distances(dist1_pdf_values, dist2_pdf_values):
    """
    Calculate KL divergence and Hellinger distance between two probability distributions.
    
    Args:
        dist1_pdf_values: PDF values of the first distribution
        dist2_pdf_values: PDF values of the second distribution
    
    Returns:
        kl_pq: KL divergence from dist1 to dist2
        kl_qp: KL divergence from dist2 to dist1
        hellinger_dist: Hellinger distance between the two distributions
    """
    # Normalize to ensure they are probability distributions
    dist1_pdf_values = dist1_pdf_values / np.sum(dist1_pdf_values)
    dist2_pdf_values = dist2_pdf_values / np.sum(dist2_pdf_values)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    dist1_pdf_values = dist1_pdf_values + epsilon
    dist2_pdf_values = dist2_pdf_values + epsilon
    
    # Calculate KL divergence
    kl_pq = entropy(dist1_pdf_values, dist2_pdf_values)
    kl_qp = entropy(dist2_pdf_values, dist1_pdf_values)
    
    # Calculate Hellinger distance
    hellinger_dist = np.sqrt(0.5 * np.sum((np.sqrt(dist1_pdf_values) - np.sqrt(dist2_pdf_values))**2))
    
    return kl_pq, kl_qp, hellinger_dist

if __name__ == "__main__":
    feature_vector_file = 'all_feature_vectors.json'  
    source_info_file = 'all_rewrite_data.json'        
    save_file = 'umap_labeled_projection.png'

    # Load data
    feature_vectors = load_feature_vectors(feature_vector_file)
    labels = load_labels_from_source(source_info_file)

    if feature_vectors is None or labels is None:
        print("Error: Failed to load data. Exiting.")
        exit()

    print("Shape of feature vectors:", feature_vectors.shape)
    print("Shape of labels:", labels.shape)
    print("First few labels:", labels[:10])

    # Check data consistency
    if feature_vectors.shape[0] != labels.shape[0]:
        raise ValueError("The number of feature vectors and labels must be the same.")

    # Standardize data
    scaled_data, scaler = standardize_data(feature_vectors)

    # Perform UMAP dimensionality reduction
    n_components_to_keep = 2 # Keep 2D embedding for easy visualization and KDE
    umap_embedding = perform_umap(scaled_data, n_components=n_components_to_keep)

    # Visualize and save
    visualize_umap_with_labels(umap_embedding, labels, save_path=save_file)

    # Calculate distribution distances
    print("\nCalculating Distribution Distances (KL Divergence & Hellinger Distance)...")

    # Separate embeddings by label
    human_embeddings = umap_embedding[labels == 0]
    gpt_embeddings = umap_embedding[labels == 1]

    print(f"Number of Human embeddings: {len(human_embeddings)}")
    print(f"Number of GPT embeddings: {len(gpt_embeddings)}")

    if len(human_embeddings) < n_components_to_keep or len(gpt_embeddings) < n_components_to_keep:
        print(f"Not enough data points (need at least {n_components_to_keep}) in one or both categories to perform KDE for distance calculations.")
    else:
        try:
            # Estimate probability density functions using KDE
            kde_human = gaussian_kde(human_embeddings.T)
            kde_gpt = gaussian_kde(gpt_embeddings.T)

            # Create evaluation grid
            x_min, y_min = np.min(umap_embedding, axis=0) - 0.1 
            x_max, y_max = np.max(umap_embedding, axis=0) + 0.1 

            # Create grid for evaluation
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