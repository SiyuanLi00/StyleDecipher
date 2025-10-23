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
        print(f"Warning: {filename} not found, loading default value.")
        return default_value
    except json.JSONDecodeError:
        print(f"Error: Unable to decode JSON from {filename}. Returning default value.")
        return default_value
    except Exception as e:
        print(f"Error loading {filename}: {e}. Returning default value.")
        return default_value

def prepare_combined_data_by_order(rewrite_data, feature_vectors):
    if len(rewrite_data) != len(feature_vectors):
        print("Error: rewrite_data and feature_vectors have mismatched lengths, cannot combine by order.")
        return []

    combined_data = []
    skipped_count = 0
    for i in range(len(rewrite_data)):
        item = rewrite_data[i]
        vector = feature_vectors[i]

        source = item.get("Source")

        if source is None:
            print(f"Warning: Skipping item at index {i} due to missing 'Source' field: {item}")
            skipped_count += 1
            continue
        
        # Ensure "human" label is 0, "GPT" label is 1
        label = 0 if source == "human" else 1 # Assume "GPT" represents machine-generated

        combined_data.append({
            "Label": label,
            "Vectors": np.array(vector) # Convert list to numpy array for consistency
        })
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} items due to missing data or format issues.")
    return combined_data

def calculate_kl_hellinger(y_true, y_pred_proba, num_bins=20):
    """
    Calculate KL divergence and Hellinger distance between Human and GPT prediction probability distributions.
    y_pred_proba is the probability that the model predicts as positive class (GPT).
    """
    # Separate Human and GPT prediction probabilities by true labels
    human_probas = y_pred_proba[y_true == 0]
    gpt_probas = y_pred_proba[y_true == 1]

    if len(human_probas) == 0 or len(gpt_probas) == 0:
        print("Warning: Unable to calculate KL divergence and Hellinger distance because one class has zero samples.")
        return None, None, None, None

    # Bin the probabilities to generate probability distributions (approximation of PMF)
    # Range is [0, 1], number of bins
    bins = np.linspace(0, 1, num_bins + 1)
    
    hist_human, _ = np.histogram(human_probas, bins=bins, density=True) # density=True normalizes to probability density
    hist_gpt, _ = np.histogram(gpt_probas, bins=bins, density=True)
    
    # Avoid log(0) problem by replacing 0 values with very small positive numbers
    epsilon = 1e-10
    P = hist_human + epsilon
    Q = hist_gpt + epsilon
    
    # Normalize to ensure they are valid probability distributions (sum to 1)
    P = P / P.sum()
    Q = Q / Q.sum()

    # Calculate KL divergence
    # D_KL(P || Q)
    kl_pq = entropy(P, Q)
    # D_KL(Q || P)
    kl_qp = entropy(Q, P)

    # Calculate Hellinger distance
    # H(P, Q) = 1/sqrt(2) * ||sqrt(P) - sqrt(Q)||_2
    hellinger_dist = np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2)) / np.sqrt(2)

    return kl_pq, kl_qp, hellinger_dist, num_bins


def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, scaler, classifier_type="xgboost"):
    """
    Train and evaluate the specified classifier (XGBoost or MLPClassifier).
    Print accuracy, classification report, F1 score, and AUROC.
    Added calculation of KL divergence and Hellinger distance between Human and GPT prediction probability distributions.
    """
    print(f"\n--- Training and Evaluating {classifier_type.upper()} ---")

    model = None
    if classifier_type == "xgboost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        X_train_scaled = X_train
        X_test_scaled = X_test
    elif classifier_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
        # Scaler should fit on training data, then be used to transform both training and test data
        X_train_scaled = scaler.fit_transform(X_train) # Each call refits, which is correct for single dataset processing
        X_test_scaled = scaler.transform(X_test)
    else:
        print("Error: Invalid classifier_type. Please choose 'xgboost' or 'mlp'.")
        return

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Get probabilities for predicting positive class (GPT)
    # predict_proba returns [probability_class0, probability_class1], we take class1 probability
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["human", "GPT"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives (human correctly classified as human): {cm[0, 0]}")
    print(f"False Positives (human incorrectly classified as GPT): {cm[0, 1]}")
    print(f"False Negatives (GPT incorrectly classified as human): {cm[1, 0]}")
    print(f"True Positives (GPT correctly classified as GPT): {cm[1, 1]}")

    # --- Add KL divergence and Hellinger distance calculation ---
    kl_pq, kl_qp, hellinger_dist, num_bins_used = calculate_kl_hellinger(y_test, y_pred_proba, num_bins=20)
    
    if kl_pq is not None:
        print(f"\n--- Prediction Probability Distribution Analysis (based on {num_bins_used} bins) ---")
        print(f"KL Divergence (Human || GPT): {kl_pq:.4f}") # Measures "information gain" from Human distribution to GPT distribution
        print(f"KL Divergence (GPT || Human): {kl_qp:.4f}") # Measures "information gain" from GPT distribution to Human distribution
        print(f"Hellinger Distance (Human, GPT): {hellinger_dist:.4f}")


# --- Data Loading ---
print("Loading data...")

# Store dataset information in a dictionary for easy iteration
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

print("Data loading completed.")

# --- Loop through each dataset and perform classification ---

for dataset_name, data_pair in datasets_info.items():
    print(f"\n{'='*50}")
    print(f"--- Processing dataset: {dataset_name} ---")
    print(f"{'='*50}")

    rewrite_data = data_pair["rewrite_data"]
    feature_vectors = data_pair["feature_vectors"]

    print(f"Dataset '{dataset_name}' rewrite data items: {len(rewrite_data)}")
    print(f"Dataset '{dataset_name}' feature vector count: {len(feature_vectors)}")

    # Prepare data for current dataset
    combined_dataset = prepare_combined_data_by_order(rewrite_data, feature_vectors)

    if not combined_dataset:
        print(f"Dataset '{dataset_name}' has no valid classification data. Skipping.")
        continue
    else:
        print(f"Successfully prepared {len(combined_dataset)} '{dataset_name}' data points.")

    # Separate features (X) and labels (y)
    # Note: Here you truncate feature vectors to the first 56 dimensions, please ensure this is your intention and all feature vectors have at least 56 dimensions.
    X = np.array([item["Vectors"] for item in combined_dataset])
    y = np.array([item["Label"] for item in combined_dataset])

    print(f"Dataset '{dataset_name}' X shape: {X.shape}")
    print(f"Dataset '{dataset_name}' y shape: {y.shape}")

    # Check if feature vector dimensions are consistent
    if X.ndim != 2 or X.shape[0] != len(combined_dataset) or X.shape[1] == 0:
        print(f"Error: Dataset '{dataset_name}' has inconsistent, empty, or incorrectly combined feature vector shapes. Skipping.")
        continue
    
    # Check if two classes exist for AUROC and KL/Hellinger calculation
    if len(np.unique(y)) < 2:
        print(f"Error: Dataset '{dataset_name}' contains only one class. Cannot calculate AUROC, KL divergence, Hellinger distance, or perform meaningful binary classification. Skipping.")
        continue

    # Split data into training and test sets
    # For each dataset, independently perform training and test set splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dataset '{dataset_name}' training set size: {len(X_train)}")
    print(f"Dataset '{dataset_name}' test set size: {len(X_test)}")

    # Re-instantiate StandardScaler for MLPClassifier each iteration
    # Because different datasets may have different statistical characteristics
    current_scaler = StandardScaler() 

    # --- Train and evaluate XGBoost ---
    train_and_evaluate_classifier(X_train, X_test, y_train, y_test, current_scaler, classifier_type="xgboost")

    # --- Train and evaluate MLPClassifier ---
    train_and_evaluate_classifier(X_train, X_test, y_train, y_test, current_scaler, classifier_type="mlp")

print("\nClassification process for all datasets completed.")