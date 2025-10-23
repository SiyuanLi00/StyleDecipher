import os
from transformers import AutoTokenizer
from models.transformer import Transformer  
import torch
from einops import rearrange, reduce
import torch.nn.functional as F
import random
from sklearn.metrics import pairwise_distances
import time


class StyleConfig:
    """Style configuration class"""
    def __init__(self, 
                 project_root=".", 
                 transformer_path="./pretrained_weights",
                 model_name="sentence-transformers/paraphrase-distilroberta-base-v1",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 max_length=512):
        self.project_root = project_root
        self.transformer_path = transformer_path
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.token_path = model_name
        
    def get_token_path(self):
        return self.token_path


class SimpleParams:
    """Simple parameters class"""
    def __init__(self):
        self.dataset_name = "raw_all"
        self.experiment_id = str(int(time.time()))
        self.version = None
        self.log_dirname = 'lightning_logs'
        self.model_type = "roberta"
        self.text_key = "syms"
        self.time_key = "hour"
        self.do_learn = False
        self.validate = False
        self.evaluate = False
        self.validate_every = 5
        self.sanity = None
        self.random_seed = 777
        self.gpus = 1
        self.period = 5
        self.suffix = ""
        
        self.learning_rate = 2e-5
        self.learning_rate_scaling = False
        self.batch_size = 128
        self.load_checkpoint = False
        self.precision = 16
        self.num_workers = 10
        self.num_epoch = 20
        self.pin_memory = False
        self.gradient_checkpointing = False
        self.temperature = 0.01
        self.multidomain_prob = None
        self.mask_bpe_percentage = 0.0
        
        self.episode_length = 16
        self.token_max_length = 32
        self.num_sample_per_author = 2
        self.embedding_dim = 512
        self.attention_fn_name = "memory_efficient"
        self.use_random_windows = False


# Default configuration instance
default_config = StyleConfig()


def create_model_and_tokenizer(config=None):
    """Create model and tokenizer"""
    if config is None:
        config = default_config
    
    params = SimpleParams()
    model = Transformer(params)
    tokenizer = AutoTokenizer.from_pretrained(config.get_token_path())
    
    return model, tokenizer, params


def reformat_tokenized_inputs(tokenized_episode):
    """Reformat tokenized inputs"""
    if len(tokenized_episode.keys()) == 3:
        input_ids, _, attention_mask = tokenized_episode.values()
        data = [input_ids, attention_mask]
    else:
        input_ids, attention_mask = tokenized_episode.values()
        data = [input_ids, attention_mask]
    return data


def preprocess_text(text, tokenizer):
    """Convert string to model input format"""
    tokenized_episode = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    
    tokenized_episode = reformat_tokenized_inputs(tokenized_episode)
    return tokenized_episode


def sample_random_window(data, window_length=32, tokenizer_param=None):
    """Sample random window, supports passing tokenizer parameter"""
    if tokenizer_param is None:
        raise ValueError("tokenizer_param is required")
        
    input_ids, attention_mask = data

    cls = tokenizer_param.cls_token_id
    pad = tokenizer_param.pad_token_id
    eos = tokenizer_param.eos_token_id
    if type(eos) != int:
        eos = tokenizer_param.sep_token_id

    # Inputs are smaller than window size -> add padding
    padding = window_length - input_ids.shape[1]
    if padding > 0:
        input_ids = F.pad(input_ids, (0, padding), 'constant', pad) 
        attention_mask = F.pad(attention_mask, (0, padding), 'constant', 0) 
        return [input_ids.unsqueeze(0).unsqueeze(0), attention_mask.unsqueeze(0).unsqueeze(0)]

    # Inputs are larger than window size -> sample random windows
    true_lengths = torch.sum(torch.where(input_ids != 1, 1, 0), 1)
    start_indices = torch.tensor([random.randint(1, l - window_length + 2) if l >= window_length else 1 for l in true_lengths])
    indices = torch.tensor([list(range(start, start + window_length - 2)) for start, l in zip(start_indices, true_lengths)])
    input_ids = input_ids.gather(1, indices)
    attention_mask = attention_mask.gather(1, indices)
        
    # Add cls token
    input_ids = F.pad(input_ids, (1, 0), 'constant', cls)
    attention_mask = F.pad(attention_mask, (1, 0), 'constant', 1)
        
    # Add eos token
    input_ids = torch.cat((input_ids, torch.where(true_lengths >= window_length, eos, pad).unsqueeze(1)), 1)
    attention_mask = torch.cat((attention_mask, torch.where(true_lengths >= window_length, 1, 0).unsqueeze(1)), 1)

    input_ids = input_ids.unsqueeze(0).unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    return [input_ids, attention_mask]
    

def mask_data_bpe(data, params_param=None, tokenizer_param=None):
    """Mask data, supports passing parameters"""
    if params_param is None:
        raise ValueError("params_param is required")
    if tokenizer_param is None:
        raise ValueError("tokenizer_param is required")
        
    if params_param.mask_bpe_percentage > 0.0:
        mask = torch.rand(data[0].size()) >= (1. - params_param.mask_bpe_percentage)

        # This is why we won't quite get to the mask percentage asked for by the user.
        pad_mask = ~(data[0] == tokenizer_param.pad_token_id)
        mask *= pad_mask

        data[0].masked_fill_(mask, tokenizer_param.mask_token_id)

    return data


def text_processing(text, tokenizer_param, params_param=None):
    """Text preprocessing"""
    if params_param is None:
        params_param = SimpleParams()
        
    # Preprocess text
    input_ids, attention_mask = preprocess_text(text, tokenizer_param)

    # Random window sampling
    if params_param.use_random_windows:
        input_ids, attention_mask = sample_random_window([input_ids, attention_mask], tokenizer_param=tokenizer_param)
    
    # Combine input_ids and attention_mask into a list
    data = [input_ids, attention_mask]

    # Mask data
    data = mask_data_bpe(data, params_param=params_param, tokenizer_param=tokenizer_param)

    input_ids = data[0].unsqueeze(0).unsqueeze(0)
    attention_mask = data[1].unsqueeze(0).unsqueeze(0)

    return [input_ids, attention_mask]


def get_final_embedding(data, model):
    """Calculate author embedding"""
    # batch_size, num_sample_per_author, episode_length
    input_ids, attention_mask = data[0], data[1]
        
    B, N, E, _ = input_ids.shape
    
    input_ids = rearrange(input_ids, 'b n e l -> (b n e) l')
    attention_mask = rearrange(attention_mask, 'b n e l -> (b n e) l')
    
    outputs = model.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True
    )

    # at this point, we're embedding individual "comments"
    comment_embeddings = model.mean_pooling(outputs['last_hidden_state'], attention_mask)
    comment_embeddings = rearrange(comment_embeddings, '(b n e) l -> (b n) e l', b=B, n=N, e=E)

    # aggregate individual comments embeddings into episode embeddings
    episode_embeddings = model.attn_fn(comment_embeddings, comment_embeddings, comment_embeddings)
    episode_embeddings = reduce(episode_embeddings, 'b e l -> b l', 'max')
        
    episode_embeddings = model.linear(episode_embeddings)
        
    return episode_embeddings, comment_embeddings


def get_all_embeddings(text, model_param, tokenizer_param, params_param):
    """Get all embeddings"""
    data = text_processing(text, tokenizer_param, params_param)
    model_param.eval()
    episode_embeddings, comment_embeddings = get_final_embedding(data, model_param)
    return episode_embeddings.cpu().detach().numpy()


def create_style_processor(config=None):
    """Factory function to create style processor"""
    if config is None:
        config = default_config
    
    model_instance, tokenizer_instance, params_instance = create_model_and_tokenizer(config)
    
    def process_text(text):
        return get_all_embeddings(text, model_instance, tokenizer_instance, params_instance)
    
    return process_text, model_instance, tokenizer_instance, params_instance


# For backward compatibility, keep global instances (but not recommended)
# Recommended to use create_style_processor() function
params = SimpleParams()
model = Transformer(params) 
tokenizer = AutoTokenizer.from_pretrained(default_config.get_token_path())


if __name__ == "__main__":
    text = "This is test text."
    
    # Method 1: Use default configuration
    processor, model_instance, tokenizer_instance, params_instance = create_style_processor()
    episode_embeddings = processor(text)
    print("Default configuration result:", episode_embeddings)
    
    # Method 2: Use custom configuration
    custom_config = StyleConfig(
        project_root=".",
        transformer_path="./pretrained_weights",
        model_name="sentence-transformers/paraphrase-distilroberta-base-v1"
    )
    processor, custom_model, custom_tokenizer, custom_params = create_style_processor(custom_config)
    custom_embeddings = processor(text)
    print("Custom configuration result:", custom_embeddings)
