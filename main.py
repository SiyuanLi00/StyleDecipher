import openai
import os
from utilities.utils import (
    load_data, get_first_1024_tokens, save_rewrite_data, 
    get_stat, xgboost_classifier, save_rewrite, save_features,
    load_json
)


class Config:
    """Configuration management class"""
    def __init__(self):
        # OpenAI configuration
        self.openai_api_key = 'YOUR_API_KEY'
        self.openai_base_url = "YOUR_URL"

        self.openai_model = "gpt-3.5-turbo"
        self.max_completion_tokens = 512
        
        # File path configuration
        self.domain_path = "dataset/att3_combine.json"
        self.rewrite_data_path = "RAID_result/att3_rewrite_data.json"
        self.feature_vectors_path = "RAID_result/att3_feature_vectors.json"
        
        # Processing parameters
        self.data_range = 2000
        
        # prompt for other text rewrite
        self.prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                           'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                           'Improve this in GPT way']
        
        # prompt list for code rewrite
        # self.prompt_list = ['Revise the code with your best effort', 'Help me polish this code', 'Rewrite the code with GPT style', 'Refine the code for me please', 'Concise the code without change the functionality']
        
        self.prefix = "\nNo need to explain. Just write code without any explanation!: "
    
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories_to_create = []
        
        # Collect all directories that need to be created
        for path in [self.rewrite_data_path, self.feature_vectors_path]:
            directory = os.path.dirname(path)
            if directory and directory not in directories_to_create:
                directories_to_create.append(directory)
        
        # Create directories
        for directory in directories_to_create:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Ensuring directory exists: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
    
    def ensure_files(self):
        """Ensure all necessary files exist, create empty files if they don't exist"""
        files_to_check = [
            (self.rewrite_data_path, []),
            (self.feature_vectors_path, [])
        ]
        
        for file_path, default_content in files_to_check:
            if not os.path.exists(file_path):
                try:
                    # Ensure directory exists
                    directory = os.path.dirname(file_path)
                    if directory:
                        os.makedirs(directory, exist_ok=True)
                    
                    # Create empty file
                    import json
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(default_content, f, ensure_ascii=False, indent=2)
                    print(f"Creating empty file: {file_path}")
                except Exception as e:
                    print(f"Error creating file {file_path}: {e}")


def setup_openai(config):
    """Setup OpenAI configuration"""
    openai.api_key = config.openai_api_key
    openai.base_url = config.openai_base_url


def use_openai_rewrite(prompt, text, config):
    """Use OpenAI for text rewriting"""
    completion = openai.chat.completions.create(
            model=config.openai_model,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}: \"{text}\"",
                },
            ],
            max_completion_tokens=config.max_completion_tokens
        )
    rewrite = completion.choices[0].message.content
    
    return rewrite


def analysis(config):
    """Main analysis function"""
    # Ensure directories and files exist
    config.ensure_directories()
    config.ensure_files()
    
    # Load data
    rewrite_data = load_json(config.rewrite_data_path, default_value=[])
    feature_vectors = load_json(config.feature_vectors_path, default_value=[])
    
    print(f"Loaded feature vectors count: {len(feature_vectors)}")
    print(f"Loaded rewrite data count: {len(rewrite_data)}")
    
    try:
        datas = load_data(config.domain_path)
        print(f"Successfully loaded dataset with {len(datas)} records")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    try:
        for idx, item in enumerate(datas[:config.data_range]):
            if (idx < len(feature_vectors)):
                print(f"Skipping already processed item {idx}")
                continue
            
            rewrite_item = {
                "Index": item["Index"],
                "Text": item["Text"],
                "Source": item["Source"]
            }
            # rewrite_item = {
            #     "Index":item["Index"],  
            #     # "Text":("Explanation: " + item["Explanation"] + "\nImplementation: " + item["Implementation"]),
            #     "Text":("Implementation: " + item["Implementation"]),
            #     "Source":item["Source"]
            # }
            rewrite_item["Text"] = get_first_1024_tokens(rewrite_item["Text"])
            print(idx)
            print(rewrite_item["Text"])
            
            for prompt in config.prompt_list:
                rewritten_text = use_openai_rewrite(prompt, rewrite_item['Text'], config)
                print(rewritten_text)
                save_rewrite_data(rewritten_text, prompt, rewrite_item)
        
            rewrite_data.append(rewrite_item)
            print("rewriting has finished!")
        
            get_stat(idx, rewrite_data, feature_vectors)
            if len(feature_vectors) == idx + 1:
                print("Feature extraction successful!")
            print(f"Feature vector: {feature_vectors[idx] if idx < len(feature_vectors) else 'N/A'}")
            
            # Save progress periodically
            if (idx + 1) % 10 == 0:
                print(f"Saving progress... (processed {idx + 1} items)")
                save_rewrite(rewrite_data, config.rewrite_data_path)
                save_features(feature_vectors, config.feature_vectors_path)
            
        print("Starting XGBoost classification...")
        xgboost_classifier(config.data_range, feature_vectors, rewrite_data)
        
        print("Saving final results...")
        save_rewrite(rewrite_data, config.rewrite_data_path)
        save_features(feature_vectors, config.feature_vectors_path)
        print("All tasks completed!")
    
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        try:
            print("Saving current progress...")
            save_rewrite(rewrite_data, config.rewrite_data_path)
            save_features(feature_vectors, config.feature_vectors_path)
            print("Progress saved successfully!")
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    return


def main():
    """Main function - controls all file I/O and configuration"""
    print("StyleDecipher starting...")
    
    # Initialize configuration
    config = Config()
    
    # Setup OpenAI
    setup_openai(config)
    
    # Run analysis
    analysis(config)
    
    print("Program ended.")


if __name__ == "__main__":
    main()