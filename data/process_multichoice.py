import json
import pandas as pd
from datasets import Dataset

# 載入 train.json, valid.json 和 context.json
with open("train.json", "r", encoding="utf-8") as train_file:
    train_data = json.load(train_file)

with open("valid.json", "r", encoding="utf-8") as valid_file:
    validation_data = json.load(valid_file)

with open("context.json", "r", encoding="utf-8") as context_file:
    context_data = json.load(context_file)

# 定義處理函數，將數字對應到 context.json 中的段落，並生成 SWAG 格式的結構
def preprocess_data(train_data, context_data):
    processed_data = []
    for item in train_data:
        question = item["question"]  # 問題
        relevant_paragraph = item["relevant"]  # 最相關的段落
        paragraphs = [context_data[p] for p in item["paragraphs"]]  # 取得所有段落
        
        # 將最相關段落的索引作為 label
        label = item["paragraphs"].index(relevant_paragraph)  # relevant 是正確答案的索引

        # 模型所需的多選格式，只選取前四個段落（如果有更多的話）
        num_choices = min(4, len(paragraphs))  # 只取前四個段落
        endings = {f"ending{i}": paragraphs[i] for i in range(num_choices)}

        # 填充到 SWAG 格式
        processed_data.append({
            "sent1": question,  # 問題
            "sent2": '',  
            **endings,          # 填入多選選項 ending0, ending1, ...
            "label": label  # 正確答案的索引
        })
    return processed_data


# 處理 train data
processed_train_data = preprocess_data(train_data, context_data)

# 將處理後的資料存成 JSON 檔案
output_file = "multiplechoice_train_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_train_data, f, ensure_ascii=False, indent=4)

# 檢查儲存的結果
print(f"Processed data saved to {output_file}")

# 處理 validation data
processed_validation_data = preprocess_data(validation_data, context_data)
output_file = "multiplechoice_validation_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_validation_data, f, ensure_ascii=False, indent=4)
print(f"Processed data saved to {output_file}")

# 將處理後的資料轉換成 Dataset 格式，方便後續進行模型訓練
dataset = Dataset.from_pandas(pd.DataFrame(processed_train_data))

# 檢查結果
print(dataset)
