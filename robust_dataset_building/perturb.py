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

from textattack.augmentation import TextBuggerAugmenter
from textattack.augmentation import TextFoolerAugmenter
from textattack.augmentation import DeepWordBugAugmenter

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


word_augmenter = TextFoolerAugmenter()

character_augmenter = DeepWordBugAugmenter()

word_character_augmenter = TextBuggerAugmenter()


def read_data(json_path):
    """
    Read data from JSON file.
    """
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def count_sentences_in_paragraph(paragraph):
    """
    Count the number of sentences in a paragraph and return the sentence list.
    """
    if not isinstance(paragraph, str):
        logging.warning(f"Input is not a string, returning empty list. Type: {type(paragraph)}")
        return []
    sentences = nltk.sent_tokenize(paragraph)
    return sentences

def apply_augmenter_to_text(text, augmenter):
    """
    Apply the specified TextAttack augmenter to the given text, processing sentence by sentence and merging results.
    """
    if not text:
        return ""

    sentences = count_sentences_in_paragraph(text)
    if not sentences:
        return "" 

    augmented_sentences = []

    for sentence in sentences:
        try:
            augmented_result = augmenter.augment(sentence)
            if augmented_result:
                augmented_sentences.append(augmented_result[0])
            else:
                augmented_sentences.append(sentence) 
        except Exception as e:
            logging.error(f"Failed to augment sentence '{sentence[:50]}...', using original sentence. Error: {e}")
            augmented_sentences.append(sentence) 

    return ' '.join(augmented_sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply TextAttack augmenters to a dataset.")
    parser.add_argument("--input_file", type=str, default="News/news_combine.json",
                        help="Path to the input JSON dataset (e.g., news_combine.json).")
    parser.add_argument("--output_file", type=str, default="News/perturbed_news_combine.json",
                        help="Path to the output JSON file for augmented data.")
    args = parser.parse_args()

    logging.info(f"Reading dataset: {args.input_file}")
    data = read_data(args.input_file)
    logging.info(f"Read {len(data)} data entries.")

    for item in tqdm(data, desc="Processing data augmentation"):
        original_text = item.get('Text') 

        if not original_text:
            logging.warning(f"Skipping entry with index {item.get('Index', 'N/A')} because 'Text' field is empty or missing.")
            continue

        logging.debug(f"Applying TextFoolerAugmenter to index {item.get('Index', 'N/A')}...")
        item['text_fooler_augmented'] = apply_augmenter_to_text(original_text, word_augmenter)
        

        logging.debug(f"Applying DeepWordBugAugmenter to index {item.get('Index', 'N/A')}...")
        item['deep_word_bug_augmented'] = apply_augmenter_to_text(original_text, character_augmenter)
        

        logging.debug(f"Applying TextBuggerAugmenter to index {item.get('Index', 'N/A')}...")
        item['text_bugger_augmented'] = apply_augmenter_to_text(original_text, word_character_augmenter)

        time.sleep(0.1) 
    
    logging.info(f"All augmenters processing completed, saving to file: {args.output_file}")

    with open(args.output_file, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logging.info("Program execution completed.")