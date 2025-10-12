import json
import os

def apply_perturbation_to_vectors(vectors, perturbation_value):
    """
    对向量列表中的每个向量的每个元素应用数值扰动。

    Args:
        vectors (list): 特征向量的列表，其中每个向量是数字列表。
        perturbation_value (float): 要加到向量每个元素上的值。

    Returns:
        list: 一个新的扰动后的特征向量列表。
    """
    perturbed_vectors = []
    for i, vector in enumerate(vectors):
        if not isinstance(vector, list):
            print(f"警告: 在索引 {i} 处发现一个非列表类型的向量: {vector}，已跳过该向量的扰动。原始向量将被保留。")
            # 根据需求，可以选择跳过、添加None或保留原始向量
            # 这里我们选择在相应的位置保留原始（或视为不可扰动的）向量/数据
            perturbed_vectors.append(vector)
            continue

        new_vector = []
        has_non_numeric = False
        for item_index, item in enumerate(vector):
            if isinstance(item, (int, float)):
                new_vector.append(item * perturbation_value)
            else:
                print(f"警告: 在原始向量索引 {i}，元素索引 {item_index} (值为: {item}) 处发现非数值元素。该元素未被扰动，将按原样保留。")
                new_vector.append(item) # 保留非数值元素
                has_non_numeric = True
        
        if has_non_numeric:
            print(f"提示: 原始向量索引 {i} (值为: {vector}) 包含非数值元素，扰动后的向量为: {new_vector}")

        perturbed_vectors.append(new_vector)
    return perturbed_vectors

def process_feature_vectors(input_file_path, output_file_path, perturbation1, perturbation2):
    """
    读取 JSON 文件中的特征向量，进行两次不同的扰动，并将原始向量和扰动后的向量保存到新的 JSON 文件中。
    假设输入文件包含一个 JSON 列表，列表中的每个元素是一个特征向量（也是一个数字列表）。

    Args:
        input_file_path (str): 包含特征向量的 JSON 文件路径。
        output_file_path (str): 保存原始和扰动后向量的 JSON 文件路径。
        perturbation1 (float): 第一次扰动的数值。
        perturbation2 (float): 第二次扰动的数值（应与第一次不同）。
    """
    original_vectors = []
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                original_vectors = data
                # 可以在这里添加更严格的检查，例如确保列表中的所有元素都是列表（向量）
                # for i, item in enumerate(original_vectors):
                #     if not isinstance(item, list):
                #         print(f"警告: 输入文件 {input_file_path} 的列表在索引 {i} 处包含非列表元素: {item}。")
                        # 根据需要决定如何处理这种情况，例如将其视为错误或尝试处理
            else:
                print(f"错误: 文件 {input_file_path} 的根元素不是列表。程序将退出。")
                return
    except FileNotFoundError:
        print(f"错误: 文件 {input_file_path} 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 {input_file_path} JSON 解码失败。")
        return
    except Exception as e:
        print(f"读取文件 {input_file_path} 时发生未知错误: {e}")
        return

    if not original_vectors and isinstance(original_vectors, list): # 允许输入是空列表[]
        print(f"文件 {input_file_path} 为空列表或未加载到向量数据。将继续处理（可能输出空扰动列表）。")
    elif not original_vectors: # original_vectors 不是列表，或者之前有错误
        print(f"未从文件 {input_file_path} 加载到向量数据，或文件内容不符合预期。")
        return


    # 应用第一次扰动 (F')
    # 我们总是基于原始向量进行扰动
    perturbed_vectors_1 = apply_perturbation_to_vectors(original_vectors, perturbation1)

    # 应用第二次扰动 (F'')
    # 同样基于原始向量进行扰动
    perturbed_vectors_2 = apply_perturbation_to_vectors(original_vectors, perturbation2)

    # 准备输出数据
    output_data = original_vectors
    output_data.extend(perturbed_vectors_1)
    output_data.extend(perturbed_vectors_2)


    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(output_data, outfile, indent=4, ensure_ascii=False)
        print(f"已成功处理向量并将结果保存到 {output_file_path}")
    except Exception as e:
        print(f"写入文件 {output_file_path} 时发生错误: {e}")

if __name__ == "__main__":
    # 1. 指定你的输入 JSON 文件路径
    # 这个文件应该包含一个特征向量的列表，例如：
    # [
    #   [1.0, 2.0, 3.0],
    #   [4.5, 5.5],
    #   [0.1, 0.2, 0.3, 0.4]
    # ]
    input_json_file = 'Abstract_result/merged_vectors.json'

    # 2. 指定合并后的输出文件名
    output_json_file = 'output_perturbed_vectors.json'

    # 3. 定义两次扰动的具体数值
    perturbation_value_1 = 0.95   # 例如，给每个特征值加 0.1
    perturbation_value_2 = 0.92 # 例如，给每个特征值减 0.05 (必须与 perturbation_value_1 不同)

    # --- (可选) 创建一个示例输入 JSON 文件 (如果它不存在) ---
    if not os.path.exists(input_json_file):
        print(f"示例输入文件 {input_json_file} 不存在，将创建一个。")
        sample_data_for_input = [
            [1.0, 2.0, 3.0],
            [10.5, 20.5, 30.5, 40.5],
            [-1.0, -2.0],
            [], # 空向量示例
            [1, 2.0, "text_element", 4], # 包含非数值元素的向量
            "not_a_vector" # 非向量元素
        ]
        try:
            with open(input_json_file, 'w', encoding='utf-8') as f_sample:
                json.dump(sample_data_for_input, f_sample, indent=4, ensure_ascii=False)
            print(f"已创建示例输入文件: {input_json_file}")
            print("示例文件内容：")
            print(json.dumps(sample_data_for_input, indent=4, ensure_ascii=False))
            print("---")
        except Exception as e:
            print(f"创建示例输入文件失败: {e}")
    # --- 示例文件创建结束 ---

    if perturbation_value_1 == perturbation_value_2:
        print("警告: 两次扰动的数值相同。F' 和 F'' 将会是一样的。请确保扰动值不同以满足需求。")

    process_feature_vectors(input_json_file, output_json_file, perturbation_value_1, perturbation_value_2)