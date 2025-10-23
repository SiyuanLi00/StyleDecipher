import numpy as np
import json
from fuzzywuzzy import fuzz
from sklearn.metrics import accuracy_score, classification_report, f1_score, pairwise_distances
from transformers import AutoModel, AutoTokenizer
import torch 
from utilities.style import get_all_embeddings, create_style_processor, StyleConfig

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
    Generate TF-IDF embedding for text using a pre-trained TfidfVectorizer.
    """
    tfidf_vector = vectorizer_obj.transform([text])
    return tfidf_vector.toarray().flatten()


# Method B: Word2Vec/GloVe average pooling embedding function
import gensim.downloader as api # Can be replaced with gensim.models.KeyedVectors to load local files

def get_word_avg_embedding(text: str, word_vectors_model) -> np.ndarray:
    """
    Use pre-trained word vector model (like GloVe/Word2Vec) for sentence average pooling.
    """
    tokens = tokenize_and_normalize(text)
    valid_vectors = [word_vectors_model[word] for word in tokens if word in word_vectors_model]
    
    if len(valid_vectors) == 0:
        # If no word is in the vocabulary, return a zero vector
        # Ensure this zero vector has the same dimension as model vectors
        if hasattr(word_vectors_model, 'vector_size'):
             return np.zeros(word_vectors_model.vector_size)
        elif hasattr(word_vectors_model, 'vector_size'): # Fallback for models without direct 'vector_size' if needed
             # This assumes all vectors in the model have the same size.
             # You might need to pick a random existing vector's size.
             # Or, if you know the dim, hardcode it.
             return np.zeros(list(word_vectors_model.values())[0].shape[0]) if len(word_vectors_model) > 0 else np.zeros(100) # Default to 100
        else: # Fallback if no vector_size attr or no vectors
             print("Warning: Unable to determine word vector dimension, returning a default-sized zero vector (dimension 100).")
             return np.zeros(100) # Assume a default dimension
    else:
        return np.mean(valid_vectors, axis=0)


# Method C: BERT/RoBERTa embedding function
def get_bert_roberta_embedding(text: str, model_obj, tokenizer_obj, pool_type: str = 'cls') -> np.ndarray:
    """
    Generate text embedding using BERT/RoBERTa model.
    Args:
        text (str): Input text.
        model_obj: Loaded BERT/RoBERTa model instance.
        tokenizer_obj: Loaded BERT/RoBERTa tokenizer instance.
        pool_type (str): Pooling type, 'cls' or 'mean'.
    Returns:
        np.ndarray: Document vector (1D).
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
        raise ValueError("pool_type must be 'cls' or 'mean'")
    
    if pooled_embedding.ndim > 1:
        pooled_embedding = pooled_embedding.squeeze()
        
    return pooled_embedding


# --- Generic feature extraction function (adjusted to accept generic embedding function and its parameters) ---
def get_feature_vector(
    item: dict,
    embedding_func: callable, # Accept a callable embedding function
    embedding_params: dict, # Accept additional parameters required by the embedding function
    cutoff_start: int,
    cutoff_end: int,
    ngram_num: int, # This parameter now represents the number of n-grams that calculate_sentence_common will return
) -> list | None:
    """
    Calculate feature vector for a single dataset item.

    Parameters:
        item (dict): An entry in the dataset, expected to contain 'Text' and possibly 'RewriteX' fields.
        embedding_func (callable): Function used to generate embeddings (like get_tfidf_embedding, get_word_avg_embedding, etc.).
        embedding_params (dict): Parameter dictionary passed to embedding_func.
        cutoff_start (int): Minimum token length threshold for original text processing.
        cutoff_end (int): Maximum token length threshold for original text processing.
        ngram_num (int): Number of n-gram features that calculate_sentence_common function will return (e.g., 4 represents 1-gram to 4-gram).

    Returns:
        list | None: If the original text of item meets length criteria, returns a numerical list containing all features;
                     otherwise returns None.
    """
    original_text = item.get('Text')
    if original_text is None:
        return None

    raw_tokens = tokenize_and_normalize(original_text)

    if len(raw_tokens) < cutoff_start or len(raw_tokens) > cutoff_end:
        return None
    
    each_data_fea = []

    # Call the passed embedding function
    raw_embedding = embedding_func(original_text, **embedding_params)

    style_features_list = [] 
    
    # Based on calculate_sentence_common behavior, its return length is fixed at 4 (1 to 4-gram)
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

            # 1. Calculate N-gram common features
            res_common = calculate_sentence_common(original_text, rewritten_text) 
            rewritten_common_features_list.extend([c / len(raw_tokens) for c in res_common]) 
            
            avg_common_features_sum = sum_for_list(avg_common_features_sum, res_common)

            # 2. Calculate fuzzy ratio
            fzwz_features_list.extend([
                fuzz.ratio(original_text, rewritten_text),
                fuzz.token_set_ratio(original_text, rewritten_text)
            ])
            
            # 3. Calculate style features (cosine distance of embeddings)
            current_embedding = embedding_func(rewritten_text, **embedding_params)
            
            cosine_dist = pairwise_distances([raw_embedding], [current_embedding], metric="cosine")[0][0]
            style_features_list.append(cosine_dist)

    # --- Merge all features ---

    # 1. Average common features
    if rewritten_count > 0:
        each_data_fea.extend([a / rewritten_count / len(raw_tokens) for a in avg_common_features_sum])
    else:
        each_data_fea.extend([0.0] * ngram_num) 

    # 2. Common features for each rewritten text
    each_data_fea.extend(rewritten_common_features_list)

    # 3. Common features between original text and all combined rewritten texts
    common_ori_vs_allcombined = calculate_sentence_common(original_text, whole_combined_text)
    each_data_fea.extend([c / len(raw_tokens) for c in common_ori_vs_allcombined])
    
    # 4. Fuzzy features for each rewritten text
    each_data_fea.extend(fzwz_features_list)
    
    # 5. Style features for each rewritten text
    each_data_fea.extend(style_features_list)
    
    return each_data_fea



def main(input_file: str, output_features_file_prefix: str, output_labels_file: str):
    """
    Main function: Load data, extract features, generate labels, and save results.
    It will loop through four different embedding methods.

    Parameters:
        input_file (str): Input file path containing original data (JSON format).
        output_features_file_prefix (str): File name prefix for saving extracted feature vectors.
                                            Actual file names will be {prefix}_{method_name}.json.
        output_labels_file (str): Output file path for saving corresponding labels (JSON format).
    """
    
    # --- Configuration parameters ---
    NGRAM_NUM = 4       # Number of n-grams returned by calculate_sentence_common (1-gram to 4-gram)
    CUTOFF_START = 5    # Minimum token length for original text
    CUTOFF_END = 20000    # Maximum token length for original text

    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            rewrite_data = json.load(f)
        print(f"Successfully loaded {len(rewrite_data)} data entries.")
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found. Please check the path.")
        return
    except json.JSONDecodeError:
        print(f"Error: Unable to decode JSON file {input_file}. Please check file format.")
        return
    except Exception as e:
        print(f"Error: Unable to load data: {e}")
        return

    # --- Pre-load and pre-train all required models/vectorizers ---

    # 1. TF-IDF Vectorizer (Method A)
    print("\n[TF-IDF] Collecting all texts to train TfidfVectorizer...")
    all_texts_for_vectorizer = []
    for item in rewrite_data:
        if 'Text' in item and item['Text'] is not None:
            all_texts_for_vectorizer.append(item['Text'])
        for key in item.keys():
            if key not in {'Text', 'common_features', 'Index', 'Source'} and item[key] is not None:
                all_texts_for_vectorizer.append(item[key])
    
    global_tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit vocabulary size
    global_tfidf_vectorizer.fit(all_texts_for_vectorizer)
    print(f"[TF-IDF] TfidfVectorizer training completed. Vocabulary size: {len(global_tfidf_vectorizer.vocabulary_)}")

    # 2. GloVe/Word2Vec model (Method B)
    print("\n[Word2Vec/GloVe] Loading pre-trained word vector model (glove-wiki-gigaword-100)...")
    global_glove_model = None
    try:
        global_glove_model = api.load("glove-wiki-gigaword-100") 
        print(f"[Word2Vec/GloVe] GloVe model loaded successfully. Vector dimension: {global_glove_model.vector_size}")
    except Exception as e:
        print(f"Error: Failed to load GloVe model. Please check network or path. Will skip this method. Error: {e}")
        # global_glove_model remains None

    # 3. BERT/RoBERTa model (Method C)
    print("\n[BERT/RoBERTa] Loading pre-trained BERT model (bert-base-uncased)...")
    global_bert_model = None
    global_bert_tokenizer = None
    bert_model_name = "bert-base-uncased" # You can choose 'roberta-base' etc.
    try:
        global_bert_model = AutoModel.from_pretrained(bert_model_name)
        global_bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        print(f"[BERT/RoBERTa] Model '{bert_model_name}' loaded successfully.")
        # Move model to GPU if available
        if torch.cuda.is_available():
            global_bert_model.to('cuda')
            print("Model moved to GPU.")
        else:
            print("GPU not available, model running on CPU.")
    except Exception as e:
        print(f"Error: Failed to load BERT model '{bert_model_name}'. Please check model name or network. Will skip this method. Error: {e}")
        # global_bert_model and global_bert_tokenizer remain None

    # 4. SBERT/BGE model (Method D) - Imported from style module, assuming they are pre-loaded
    print("\n[SBERT/BGE] Will use model and tokenizer imported from 'style' module.")
    # If style.model or style.tokenizer is None, should also check here and skip
    if model is None or tokenizer is None:
        print("Warning: style module's SBERT/BGE model or tokenizer not loaded correctly. SBERT/BGE method will be skipped.")
        sbert_bge_available = False
    else:
        sbert_bge_available = True
    sbert_bge_available = False
    # Define the set of embedding methods to test
    # Structure: "method_name": (embedding_function, parameter_dictionary_required_by_embedding_function)
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
        print("No available embedding methods. Please check model loading status.")
        return

    # Loop through different method experiments
    # Label files only need to be saved once, as they are independent of feature extraction methods
    # In this loop, all_labels will be generated from the last iteration, all labels should be the same.
    # To avoid duplication, labels can be generated and saved separately outside the loop.
    first_run = True # Mark if it's the first loop iteration, used for saving labels
    
    for method_name, (embedding_func, embedding_params) in experiment_setups.items():
        print(f"\n--- Starting feature extraction for {method_name} method ---")
        current_feature_vectors = []
        current_labels = [] # Generate a set of labels for each loop to ensure correspondence with current_feature_vectors

        processed_count = 0
        skipped_count = 0

        for i, item in enumerate(rewrite_data):
            if (i + 1) % 100 == 0:
                print(f"  [{method_name}] Processed {i + 1}/{len(rewrite_data)} entries...")

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
                # Label generation logic (method-independent)
                if item.get("Source") == "human":
                    current_labels.append(0) # Human label is 0
                else:
                    current_labels.append(1) # Other (like GPT) label is 1
                processed_count += 1
            else:
                skipped_count += 1
        
        print(f"\n[{method_name}] Feature extraction completed.")
        print(f"  Successfully processed entries: {processed_count}")
        print(f"  Skipped entries: {skipped_count}")
        print(f"  Total {method_name} feature vectors: {len(current_feature_vectors)}")

        if not current_feature_vectors:
            print(f"[{method_name}] No feature vectors extracted. Skipping save.")
            continue

        # Save feature vectors (use different file names for different methods)
        current_output_features_file = f"{output_features_file_prefix}_{method_name}.json"
        print(f"Saving {method_name} feature vectors to {current_output_features_file}...")
        try:
            # Convert NumPy arrays to lists for JSON serialization
            serializable_feature_vectors = [vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in current_feature_vectors]
            with open(current_output_features_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_feature_vectors, f, ensure_ascii=False, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
            print(f"Feature vectors saved to {current_output_features_file}")
        except Exception as e:
            print(f"Error occurred while saving {method_name} feature vectors: {e}")

        # Label files only need to be saved once, as they are the same for all methods
        if first_run:
            print(f"\nSaving labels to {output_labels_file}...")
            try:
                with open(output_labels_file, 'w', encoding='utf-8') as f:
                    json.dump(current_labels, f, ensure_ascii=False, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
                print(f"Labels saved to {output_labels_file}")
                first_run = False # Mark as saved
            except Exception as e:
                print(f"Error occurred while saving labels: {e}")

# --- Program entry point ---
if __name__ == "__main__":
    # Define input and output file paths
    INPUT_DATA_FILE = 'Yelp_result/rewrite_data.json' # Original dataset file
    OUTPUT_FEATURES_FILE_PREFIX = 'Yelp_result/all_feature_vectors' # File prefix for saving extracted features
    OUTPUT_LABELS_FILE = 'Yelp_result/all_labels.json' # File for saving corresponding labels

    main(INPUT_DATA_FILE, OUTPUT_FEATURES_FILE_PREFIX, OUTPUT_LABELS_FILE)