# StyleDecipher: Robust and Explainable Detection of LLM-Generated Texts with Stylistic Analysis



## Overview

StyleDecipher is a robust and explainable detection framework that revisits LLM generated text detection using combined feature extractors to quantify stylistic differences. 



## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for optimal performance)


### Setup

1. Clone the repository:
```bash
git clone https://github.com/SiyuanLi00/StyleDecipher.git
cd StyleDecipher
```

2. Create a virtual environment:
```bash
conda create -n style_decipher python=3.9
conda activate style_decipher
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```



## Quick Start

### Basic Usage
Here we will show you how to use StyleDecipher to extract style features from a dataset. This is also the In-Domain Experiment in the paper.

1. **Prepare your dataset** in the required JSON format:
```json
[
    {
        "Index": 1,
        "Text": "Your text content here...",
        "Source": "human"  // or "GPT" for machine-generated
    }
]
```

1. **Run style extraction and analysis**:
Set up your config in `main.py` and run `main.py` to extract style features, including openai api key, base url, model name, and file path configuration.
```python
class Config:
    """Configuration management class"""
    def __init__(self):
        # OpenAI configuration
        self.openai_api_key = 'YOUR_API_KEY'
        self.openai_base_url = "YOUR_URL"

        self.openai_model = "REWRITE_MODEL"
        self.max_completion_tokens = 512
        
        # File path configuration
        self.domain_path = "your/dataset/path.json"
        self.rewrite_data_path = "your/rewrite_data_path.json"
        self.feature_vectors_path = "your/feature_vectors_path.json"

```
```bash
python main.py
```

After building the feature vectors, you can also run 'experiment_main.py' to perform classification.

```bash
python experiment_main.py
```

## Experiment

Here we will discuss how to run other experiments mentioned in the paper.

### Out-of-Domain Experiment
To run the Out-of-Domain Experiment, you need to prepare enough domain datasets to cover all the domains you want to evaluate. And run style extraction and analysis by `main.py` with different dataset path setting.

After that, setting your style extraction results in `experiment_OOD.py` and run it to evaluate the StyleDecipher on out-of-domain datasets.
```bash
python experiment_OOD.py
```

### Robustness Experiment
You can use python scripts in `robust_dataset_building/` to build robustness datasets. And evaluate them by `main.py` with same approach as In-Domain Experiment.

### Pluggable Performance
To evaluate the pluggable performance of StyleDecipher, you can run `experiment_plugin.py` to extract features using different text embedding method after getting rewritten data and feature vectors in `main.py`. 

### Explainability Evaluation and Visualization
To evaluate the explainability of StyleDecipher, you can run scripts in `plot/` to visualize data distribution and get explanation metric results.












