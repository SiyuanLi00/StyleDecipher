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

default_config = StyleConfig()

class SimpleParams:
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

def create_model_and_tokenizer(config=None):
    if config is None:
        config = default_config
    
    params = SimpleParams()
    model = Transformer(params)
    tokenizer = AutoTokenizer.from_pretrained(config.get_token_path())
    
    return model, tokenizer, params

params = SimpleParams()
model = Transformer(params) 
tokenizer = AutoTokenizer.from_pretrained(default_config.get_token_path())

def reformat_tokenized_inputs(tokenized_episode):
    if len(tokenized_episode.keys()) == 3:
        input_ids, _, attention_mask = tokenized_episode.values()
        data = [input_ids, attention_mask]
    else:
        input_ids, attention_mask = tokenized_episode.values()
        data = [input_ids, attention_mask]
    return data

def preprocess_text(text, tokenizer):
    """将字符串转换为模型输入格式"""
    tokenized_episode = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    
    tokenized_episode = reformat_tokenized_inputs(tokenized_episode)
    return tokenized_episode

def sample_random_window(data, window_length=32, tokenizer_param=None):
    """采样随机窗口，支持传入tokenizer参数"""
    if tokenizer_param is None:
        # 使用全局tokenizer作为后备
        tokenizer_param = tokenizer
        
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
    """掩码数据，支持传入参数"""
    if params_param is None:
        params_param = params
    if tokenizer_param is None:
        tokenizer_param = tokenizer
        
    if params_param.mask_bpe_percentage > 0.0:
        mask = torch.rand(data[0].size()) >= (1. - params_param.mask_bpe_percentage)

        # This is why we won't quite get to the mask percentage asked for by the user.
        pad_mask = ~(data[0] == tokenizer_param.pad_token_id)
        mask *= pad_mask

        data[0].masked_fill_(mask, tokenizer_param.mask_token_id)

    return data

def text_processing(text, tokenizer_param, params_param=None):
    """文本预处理"""
    if params_param is None:
        params_param = params
        
    # 预处理文本
    input_ids, attention_mask = preprocess_text(text, tokenizer_param)

    # 随机窗口采样
    if params_param.use_random_windows:
        input_ids, attention_mask = sample_random_window([input_ids, attention_mask], tokenizer_param=tokenizer_param)
    
    # 将 input_ids 和 attention_mask 组合成列表
    data = [input_ids, attention_mask]

    # 掩码数据
    data = mask_data_bpe(data, params_param=params_param, tokenizer_param=tokenizer_param)

    input_ids = data[0].unsqueeze(0).unsqueeze(0)
    attention_mask = data[1].unsqueeze(0).unsqueeze(0)

    return [input_ids, attention_mask]

def get_final_embedding(data, model):
    """Computes the Author Embedding. 
    """
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

def get_all_embeddings(text, model_param=None, tokenizer_param=None, params_param=None):
    """获取所有嵌入，支持传入自定义模型和分词器"""
    if model_param is None:
        model_param = model
    if tokenizer_param is None:
        tokenizer_param = tokenizer
    if params_param is None:
        params_param = params
        
    data = text_processing(text, tokenizer_param, params_param)
    model_param.eval()
    episode_embeddings, comment_embeddings = get_final_embedding(data, model_param)
    return episode_embeddings.cpu().detach().numpy()

def create_style_processor(config=None):
    """创建样式处理器的工厂函数"""
    if config is None:
        config = default_config
    
    model_instance, tokenizer_instance, params_instance = create_model_and_tokenizer(config)
    
    def process_text(text):
        return get_all_embeddings(text, model_instance, tokenizer_instance, params_instance)
    
    return process_text, model_instance, tokenizer_instance, params_instance

if __name__ == "__main__":
    text = "This is test text."
    
    # 方式1：使用默认配置
    episode_embeddings = get_all_embeddings(text)
    print("默认配置结果:", episode_embeddings)
    
    # 方式2：使用自定义配置
    custom_config = StyleConfig(
        project_root=".",
        transformer_path="./pretrained_weights",
        model_name="sentence-transformers/paraphrase-distilroberta-base-v1"
    )
    processor, custom_model, custom_tokenizer, custom_params = create_style_processor(custom_config)
    custom_embeddings = processor(text)
    print("自定义配置结果:", custom_embeddings)
