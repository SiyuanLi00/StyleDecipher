import os
from transformers import AutoTokenizer
from models.transformer import Transformer  
from arguments import create_argument_parser  
import torch
from einops import rearrange, reduce
import torch.nn.functional as F
import random
from utilities.file_utils import Utils as utils
from sklearn.metrics import pairwise_distances

params = create_argument_parser()
model = Transformer(params) 
token_path = os.path.join(utils.transformer_path, "paraphrase-distilroberta-base-v1")
tokenizer = AutoTokenizer.from_pretrained(token_path)  

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

def sample_random_window(data, window_length=32):
    input_ids, attention_mask = data

    cls = tokenizer.cls_token_id
    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    if type(eos) != int:
        eos = tokenizer.sep_token_id

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

    # 确保返回的数据是 4 维的
    input_ids = input_ids.unsqueeze(0).unsqueeze(0)  # 添加 batch_size 和 num_sample_per_author 维度
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    return [input_ids, attention_mask]
    
    

def mask_data_bpe(data):
    if params.mask_bpe_percentage > 0.0:
        mask = torch.rand(data[0].size()) >= (1. - params.mask_bpe_percentage)

        # This is why we won't quite get to the mask percentage asked for by the user.
        pad_mask = ~(data[0] == tokenizer.pad_token_id)
        mask *= pad_mask

        data[0].masked_fill_(mask, tokenizer.mask_token_id)

    return data

def text_processing(text, tokenizer):
    """文本预处理"""
    # 预处理文本
    input_ids, attention_mask = preprocess_text(text, tokenizer)

    # 随机窗口采样
    if params.use_random_windows:
        input_ids, attention_mask = sample_random_window([input_ids, attention_mask])
    
    # 将 input_ids 和 attention_mask 组合成列表
    data = [input_ids, attention_mask]

    # 掩码数据
    data = mask_data_bpe(data)

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

def get_all_embeddings(text, model, tokenizer):
    data = text_processing(text, tokenizer)
    model.eval()
    episode_embeddings, comment_embeddings = get_final_embedding(data, model)
    return episode_embeddings.cpu().detach().numpy()



if __name__ == "__main__":
    text = "This is test text."
    model = Transformer(params) 
    episode_embeddings = get_all_embeddings(text, model, tokenizer)
    
    print(episode_embeddings)
