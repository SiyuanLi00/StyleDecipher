import numpy as np
import json
import os
from fuzzywuzzy import fuzz
import tiktoken
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, pairwise_distances
from .style import get_all_embeddings, create_style_processor, StyleConfig


def ensure_directory_exists(file_path):
    """Ensure the directory of the file path exists"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Creating directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")


def load_json(filename, default_value):
    """Load JSON file with support for automatic empty file creation"""
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"{filename} does not exist, creating default file.")
            ensure_directory_exists(filename)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(default_value, f, ensure_ascii=False, indent=2)
            return default_value
            
        # Try to read file
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:  # File is empty
                print(f"{filename} is empty, returning default value.")
                return default_value
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"{filename} JSON format error: {e}, returning default value.")
        return default_value
    except Exception as e:
        print(f"Error loading {filename}: {e}, returning default value.")
        return default_value


def get_first_1024_tokens(text):
    """Get the first 1024 tokens of text"""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        first_1024_tokens = tokens[:1024]
        decoded_tokens = enc.decode(first_1024_tokens)
        return decoded_tokens
    except Exception as e:
        print(f"Error processing tokens: {e}")
        return text[:4000]  # Return first 4000 characters as fallback


def tokenize_and_normalize(sentence):
    """Tokenize and normalize sentence"""
    return [word.lower().strip() for word in sentence.split()]


def extract_ngrams(tokens, n):
    """Extract n-grams"""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def common_elements(list1, list2):
    """Find common elements between two lists"""
    return set(list1) & set(list2)


def calculate_sentence_common(sentence1, sentence2):
    """Calculate similarity between two sentences"""
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)
    
    unigrams1 = extract_ngrams(tokens1, 1)
    unigrams2 = extract_ngrams(tokens2, 2)
    
    bigrams1 = extract_ngrams(tokens1, 2)
    bigrams2 = extract_ngrams(tokens2, 2)
    
    common_unigrams = len(common_elements(unigrams1, unigrams2))
    common_bigrams = len(common_elements(bigrams1, bigrams2))
    
    return common_unigrams, common_bigrams


def sum_for_list(a, b):
    """Sum elements of lists"""
    return [x + y for x, y in zip(a, b)]


def get_stat(index: int, rewrite_data: list, feature_vectors: list):
    """Get statistics and calculate feature vectors"""
    try:
        # Create style processor
        processor, model_instance, tokenizer_instance, params_instance = create_style_processor()
        
        rewrite_item = rewrite_data[index]
        original_text = rewrite_item["Text"]
        
        # Get original text embedding
        original_embedding = processor(original_text)
        
        # Calculate features for all rewritten versions
        all_features = []
        
        # Define prompt list (consistent with main.py)
        prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                      'Make this fluent while doing minimal change', 'Refine this for me please', 
                      'Concise this for me and keep all the information', 'Improve this in GPT way']
        
        for prompt in prompt_list:
            if prompt in rewrite_item:
                rewritten_text = rewrite_item[prompt]
                
                # Get rewritten text embedding
                rewritten_embedding = processor(rewritten_text)
                
                # Calculate distance features
                euclidean_distance = pairwise_distances(original_embedding, rewritten_embedding, metric='euclidean')[0][0]
                cosine_distance = pairwise_distances(original_embedding, rewritten_embedding, metric='cosine')[0][0]
                manhattan_distance = pairwise_distances(original_embedding, rewritten_embedding, metric='manhattan')[0][0]
                
                # Calculate text similarity features
                common_unigrams, common_bigrams = calculate_sentence_common(original_text, rewritten_text)
                fuzz_ratio = fuzz.ratio(original_text, rewritten_text)
                
                # Combine features
                features = [euclidean_distance, cosine_distance, manhattan_distance, 
                           common_unigrams, common_bigrams, fuzz_ratio]
                all_features.append(features)
        
        # Calculate average features or other aggregation methods
        if all_features:
            # Calculate average features across all prompts
            avg_features = [sum(feature_list) / len(feature_list) for feature_list in zip(*all_features)]
            feature_vectors.append(avg_features)
        else:
            # If no rewritten text, use zero features
            feature_vectors.append([0.0] * 6)
            
    except Exception as e:
        print(f"Error calculating features (index {index}): {e}")
        # Add default feature vector
        feature_vectors.append([0.0] * 6)


def load_data(address):
    """Load data file with better error handling"""
    try:
        if not os.path.exists(address):
            raise FileNotFoundError(f"Data file does not exist: {address}")
            
        with open(address, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Data file is empty: {address}")
            datas = json.loads(content)
            
        if not isinstance(datas, list):
            raise ValueError(f"Data file format error, should be a list: {address}")
            
        return datas
    except Exception as e:
        print(f"Failed to load data file: {e}")
        raise


def save_rewrite_data(rewritten_text, prompt, rewrite_item):
    """Save rewrite data"""
    rewrite_item[prompt] = rewritten_text


def xgboost_classifier(data_range: int, feature_vectors: list, rewrite_data: list):
    """XGBoost classifier"""
    try:
        if not feature_vectors or not rewrite_data:
            print("Warning: Feature vectors or rewrite data is empty, skipping classification")
            return
            
        # Ensure data range doesn't exceed actual data amount
        actual_range = min(data_range, len(feature_vectors), len(rewrite_data))
        
        stack_feature_vectors = np.vstack(feature_vectors[:actual_range])
        labels = np.array([0 if _["Source"] == "human" else 1 for _ in rewrite_data[:actual_range]])
        
        if len(set(labels)) < 2:
            print("Warning: Insufficient label categories, skipping classification")
            return
            
        x_train, x_test, y_train, y_test = train_test_split(
            stack_feature_vectors, labels, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        clf = MLPClassifier(
            hidden_layer_sizes=(10,),
            max_iter=1000, 
            activation='relu', 
            solver='adam',   
            random_state=42  
        )
        
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        
        print("Accuracy:", accuracy_score(y_test, y_pred), "F1 score", f1_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))
        
    except Exception as e:
        print(f"Error during classification: {e}")


def save_rewrite(rewrite_data, filename):
    """Save rewrite data to file with automatic directory creation"""
    try:
        ensure_directory_exists(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(rewrite_data, f, ensure_ascii=False, indent=4)
        print(f"Rewrite text saved to {filename}!")
    except Exception as e:
        print(f"Failed to save rewrite data: {e}")


def save_features(feature_vectors, filename):
    """Save feature vectors to file with automatic directory creation"""
    try:
        ensure_directory_exists(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(feature_vectors, f, ensure_ascii=False, indent=4, 
                     default=lambda x: float(x) if isinstance(x, np.float32) else x)
        print(f"Feature vectors saved to {filename}!")
    except Exception as e:
        print(f"Failed to save feature vectors: {e}")


def save_json(data, filename):
    """Save data as JSON file with automatic directory creation"""
    try:
        ensure_directory_exists(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4, 
                     default=lambda x: float(x) if isinstance(x, np.float32) else x)
        print(f"data saved to {filename}")
    except Exception as e:
        print(f"save {filename} failed: {e}")
