import csv
import argparse
from transformers import BertTokenizer, BertForMultipleChoice, pipeline
import torch
import json
from tqdm import tqdm

# 設定命令行參數
def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple-choice and QA model inference")
    parser.add_argument("context_file", type=str, help="Path to context JSON file")
    parser.add_argument("test_file", type=str, help="Path to test JSON file")
    parser.add_argument("output_file", type=str, help="Path to save the prediction CSV file")
    return parser.parse_args()

def main():
    args = parse_args()
    multiple_choice_model_path = "./unzipped/output_multiple_choice"
    QA_model_path = "./unzipped/output_QA"

    multiple_choice_model =  BertForMultipleChoice.from_pretrained(multiple_choice_model_path)
    multiple_choice_tokenizer = BertTokenizer.from_pretrained(multiple_choice_model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    multiple_choice_model.to(device)  # 將多選題模型移動到 GPU

    QA_model = pipeline("question-answering", model=QA_model_path, device=0)


    with open(args.context_file, "r", encoding="utf-8") as context_file:
        context_data = json.load(context_file)

    with open(args.test_file, "r", encoding="utf-8") as test_file:
        test_data = json.load(test_file)


    # Get the model's maximum position embeddings size
    max_length = multiple_choice_model.config.max_position_embeddings

    result_list = []
    # 處理輸入並進行推理
    for data in tqdm(test_data):
        question = data["question"]
        options = [context_data[data["paragraphs"][i]] for i in range(4)]

        # 將問題和每個選項進行_multiple_choice組合並編碼
        encodings = multiple_choice_tokenizer(
            [question] * len(options),  # 重複問題
            options,                    # 每個選項
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # 模型推理
        outputs = multiple_choice_model(**{k: v.unsqueeze(0).to(device) for k, v in encodings.items()})
        
        # 獲取 logits 值
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        # 印每個選項的概率
        # for i, prob in enumerate(probabilities):
        #     print(f"選項 {i} 概率: {prob.item():.4f}")

        # 選擇概率最高的選項
        best_option = torch.argmax(probabilities).item()
        # print(f"模型認為最佳選項是: 選項 {best_option}")

        # 問答模型
        question = data["question"]
        context = context_data[data["paragraphs"][best_option]]
        result = QA_model(question=question, context=context)
        # print(f"問答模型回答: {result['answer']}")
        result_list.append([data['id'] ,result['answer']])

    # 將結果寫入csv

    with open(args.output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "answer"]) # 寫入 CSV 標題行
        writer.writerows(result_list)  # 寫入結果行
        print("success")

if __name__ == "__main__":
    main()
# $kaggle competitions submit -c ntu-adl-2024-hw-1-chinese-extractive-qa -f submission.csv -m "Message"

    