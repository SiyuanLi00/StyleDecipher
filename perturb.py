import argparse
import json
import logging
import random
import re
import sys
import time
import nltk
import numpy as np
import torch
from tqdm import tqdm

# 从 textattack.augmentation 导入增强器
from textattack.augmentation import TextBuggerAugmenter
from textattack.augmentation import TextFoolerAugmenter
from textattack.augmentation import DeepWordBugAugmenter

# 确保 NLTK tokenizer 数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 初始化增强器实例
# TextFoolerAugmenter 是一种词替换增强器
word_augmenter = TextFoolerAugmenter()
# DeepWordBugAugmenter 是一种字符级增强器，引入拼写错误等
character_augmenter = DeepWordBugAugmenter()
# TextBuggerAugmenter 是一种结合词和字符级修改的增强器
word_character_augmenter = TextBuggerAugmenter()


def read_data(json_path):
    """
    从 JSON 文件中读取数据。
    """
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def count_sentences_in_paragraph(paragraph):
    """
    计算段落中的句子数量并返回句子列表。
    """
    if not isinstance(paragraph, str):
        logging.warning(f"Input is not a string, returning empty list. Type: {type(paragraph)}")
        return []
    sentences = nltk.sent_tokenize(paragraph)
    return sentences

def apply_augmenter_to_text(text, augmenter):
    """
    对给定的文本应用指定的TextAttack增强器，逐句处理并合并结果。
    """
    if not text:
        return ""

    sentences = count_sentences_in_paragraph(text)
    if not sentences:
        return "" # 如果没有句子（例如，空字符串），则返回空

    augmented_sentences = []
    # TextAttack的增强器通常设计为处理单个文本字符串。
    # 您提供的用法 `augmenter.augment(d)[0]` 表明您希望逐句处理。
    # 因此，我们这里将文本分割成句子，然后对每个句子进行增强。
    for sentence in sentences:
        try:
            # augmenter.augment 方法期望一个字符串，并返回一个增强后的字符串列表
            # 我们取第一个结果，这在默认情况下是TextAttack增强器的常见行为
            augmented_result = augmenter.augment(sentence)
            if augmented_result:
                augmented_sentences.append(augmented_result[0])
            else:
                augmented_sentences.append(sentence) # 如果增强失败，保留原句
        except Exception as e:
            logging.error(f"增强句子 '{sentence[:50]}...' 失败，使用原始句子。错误: {e}")
            augmented_sentences.append(sentence) # 出现错误时，保留原句

    # 将所有增强后的句子重新合并成一个段落
    return ' '.join(augmented_sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply TextAttack augmenters to a dataset.")
    parser.add_argument("--input_file", type=str, default="News/news_combine.json",
                        help="Path to the input JSON dataset (e.g., news_combine.json).")
    parser.add_argument("--output_file", type=str, default="News/perturbed_news_combine.json",
                        help="Path to the output JSON file for augmented data.")
    args = parser.parse_args()

    logging.info(f"正在读取数据集: {args.input_file}")
    data = read_data(args.input_file)
    logging.info(f"已读取 {len(data)} 条数据。")

    # 遍历数据集并应用增强
    for item in tqdm(data, desc="处理数据增强"):
        original_text = item.get('Text') # 获取原始文本

        if not original_text:
            logging.warning(f"跳过索引为 {item.get('Index', 'N/A')} 的条目，因为 'Text' 字段为空或缺失。")
            continue

        # 应用 TextFoolerAugmenter (词替换)
        logging.debug(f"正在应用 TextFoolerAugmenter 到索引 {item.get('Index', 'N/A')}...")
        item['text_fooler_augmented'] = apply_augmenter_to_text(original_text, word_augmenter)
        
        # 应用 DeepWordBugAugmenter (字符级错误)
        logging.debug(f"正在应用 DeepWordBugAugmenter 到索引 {item.get('Index', 'N/A')}...")
        item['deep_word_bug_augmented'] = apply_augmenter_to_text(original_text, character_augmenter)
        
        # 应用 TextBuggerAugmenter (词+字符级混合)
        logging.debug(f"正在应用 TextBuggerAugmenter 到索引 {item.get('Index', 'N/A')}...")
        item['text_bugger_augmented'] = apply_augmenter_to_text(original_text, word_character_augmenter)

        # 为了避免短时间内频繁调用导致潜在问题或资源耗尽，可以加入少量延迟
        time.sleep(0.1) 
    
    logging.info(f"所有增强器处理完成，正在保存到文件: {args.output_file}")
    # 将更新后的数据保存到新的 JSON 文件
    with open(args.output_file, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logging.info("程序执行完毕。")