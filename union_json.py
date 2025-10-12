import json
import os

def merge_json_files(input_file_paths, output_file_path):
    """
    合并多个结构相同的 JSON 文件的内容到一个新的 JSON 文件中。
    假设每个输入文件包含一个 JSON 列表。

    Args:
        input_file_paths (list): 包含要合并的 JSON 文件路径的列表。
        output_file_path (str): 合并后的内容要保存到的 JSON 文件路径。
    """
    all_entries = []
    for file_path in input_file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_entries.extend(data)
                else:
                    print(f"警告: 文件 {file_path} 的根元素不是列表，已跳过。")
        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 未找到。")
        except json.JSONDecodeError:
            print(f"错误: 文件 {file_path} JSON 解码失败。")

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(all_entries, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # 替换为你要合并的 JSON 文件路径列表
    files_to_merge = [
        'Yelp_result/rewrite_data.json',
        'News_result/rewrite_data.json',
        'Essay_result/rewrite_data.json',
    ]
    merged_file = 'all_rewrite_data.json'  # 指定合并后的输出文件名

    merge_json_files(files_to_merge, merged_file)
    print(f"已成功将文件合并到 {merged_file}")