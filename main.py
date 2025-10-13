from utils import *
import openai


openai.api_key = 'YOUR_API_KEY'
openai.base_url = "YOUR_URL"


prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                'Improve this in GPT way']

# prompt list for code rewrite
# prompt_list = ['Revise the code with your best effort', 'Help me polish this code', 'Rewrite the code with GPT style', 'Refine the code for me please', 'Concise the code without change the functionality']

prefix = "\nNo need to explain. Just write code without any explanation!: "


def use_openai_rewrite(prompt, text):
    completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}: \"{text}\"",
                },
            ],
            max_completion_tokens=512
        )
    rewrite = completion.choices[0].message.content
    
    return rewrite


def analysis():
    print(len(feature_vectors))
    data_range = 2000
    datas = load_data(DOMAIN_PATH)
    try:
        for idx, item in enumerate(datas[: data_range]):
            if (idx < len(feature_vectors)):
                continue
            
            rewrite_item = {
                "Index":item["Index"],
                "Text":item["Text"],
                "Source":item["Source"]
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
            for prompt in prompt_list:
                rewritten_text = use_openai_rewrite(prompt, rewrite_item['Text'])
                print(rewritten_text)
                save_rewrite_data(rewritten_text, prompt, rewrite_item)
        
            rewrite_data.append(rewrite_item)
            print("rewriting has finished!")
        
            get_stat(idx)
            if len(feature_vectors) == idx + 1:
                print("estimate call successed!")
            print(feature_vectors[idx])
            
        xgboost_classifier(data_range)
        save_rewrite(rewrite_data)
        save_features(feature_vectors)
    
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        save_rewrite(rewrite_data, REWRITE_DATA_PATH)
        save_features(feature_vectors, FEATURE_VECTORS_PATH)
    
    return


analysis()