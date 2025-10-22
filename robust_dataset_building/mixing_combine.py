import json
import random

def read_data(json_path):
    """
    Reads a JSON file and returns its content.
    """
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def combine_and_reindex_data(human_file_path, gpt_file_path, output_file_path):
    """
    Reads two JSON files, takes the smaller count of items from each,
    re-indexes them, and saves to a new JSON file.

    Args:
        human_file_path (str): Path to the human-generated data JSON file (e.g., 'yelp_human.json').
        gpt_file_path (str): Path to the GPT-generated data JSON file (e.g., 'yelp_mixed_data_GPT.json').
        output_file_path (str): Path for the combined and re-indexed output JSON file.
    """
    # 1. Read data from both files
    human_data = read_data(human_file_path)
    gpt_data = read_data(gpt_file_path)

    print(f"Read {len(human_data)} items from {human_file_path}")
    print(f"Read {len(gpt_data)} items from {gpt_file_path}")

    # 2. Determine the smaller quantity of items
    min_items = min(len(human_data), len(gpt_data))
    print(f"Smallest item count is: {min_items}")

    # 3. Randomly sample 'min_items' from each list
    random.seed(2023) # For reproducibility
    sampled_human_data = random.sample(human_data, min_items)
    sampled_gpt_data = random.sample(gpt_data, min_items)

    # 4. Combine the sampled data
    combined_data = sampled_human_data + sampled_gpt_data
    random.shuffle(combined_data) # Shuffle the combined list to mix human and GPT entries

    # 5. Re-index the combined data
    reindexed_data = []
    for i, item in enumerate(combined_data):
        # Create a new dictionary to avoid modifying the original items in place
        # and to ensure only specified keys are carried over
        
        new_item = {
            "Index": i, # Assign new sequential index
            "Text": item.get("Explanation", "") + item.get("Implementation", "") or item.get("Text", ""), # Keep original text
            "Source": item.get("Source", "unknown") # Keep original source (human or GPT)
        }
        reindexed_data.append(new_item)

    # 6. Write the re-indexed data to the new JSON file
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(reindexed_data, f, ensure_ascii=False, indent=4)

    print(f"Successfully combined and re-indexed {len(reindexed_data)} items to {output_file_path}")

# --- How to use this function ---
# Make sure you have 'yelp_human.json' and 'yelp_mixed_data_GPT.json' in the same directory.
# You can create dummy files for testing if you don't have them yet.


# Call the function with your file paths
combine_and_reindex_data(
    'dataset/HumanEval Code/code_human.json',
    'dataset/HumanEval Code/code_mixed_data_GPT.json',
    'dataset/HumanEval Code/code_mixed_data_combine.json'
)