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

# 定義處理函數，將數字對應到 context.json 中的段落
def preprocess_data(train_data, context_data):
    processed_data = []
    for item in train_data:
        question = item["question"]  # 問題
        context = context_data[item["relevant"]]  # 最相關的段落
        answer = item["answer"]
        id = item["id"]

        # 檢查 answer_start 和 text 是否是列表，若不是則轉換為列表
        if not isinstance(answer["start"], list):
            answer["start"] = [answer["start"]]
        if not isinstance(answer["text"], list):
            answer["text"] = [answer["text"]]


        # 填充到 SQuAD 格式
        processed_data.append({
            "id": id,
            "question": question,  # 問題
            "context": context,  # 最相關的段落
            "answers": {  # 正確答案
                "text": answer["text"],  # 答案文本
                "answer_start": answer["start"]  # 答案的起始位置
            }
        })
    return processed_data


# 處理 train data
processed_train_data = preprocess_data(train_data, context_data)

# 將處理後的資料存成 JSON 檔案
output_file = "QA_train_data.json"
with open(output_file, "w", encoding="utf-8") as f:
     json.dump({"data": processed_train_data}, f, ensure_ascii=False, indent=4)

# 檢查儲存的結果
print(f"Processed data saved to {output_file}")

# 處理 validation data
processed_validation_data = preprocess_data(validation_data, context_data)
output_file = "QA_validation_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"data": processed_validation_data}, f, ensure_ascii=False, indent=4)
print(f"Processed data saved to {output_file}")

# 將處理後的資料轉換成 Dataset 格式，方便後續進行模型訓練
dataset = Dataset.from_pandas(pd.DataFrame(processed_train_data))

# 檢查結果
print(dataset)
# Dataset({
#     features: ['id', 'question', 'context', 'answers'],
#     num_rows: 21714
# })