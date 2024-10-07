## Q1: Data processing
### Tokenizer
- 使用了Chinese Word Segmentation (CWS) tool LTP將句子分成幾個詞，認出中文詞中的boumdary，此tokenizer只會影響pretraining中的masking詞為何，不會在fine tuning 中作用，用來達成中文中的wwm
	- 如`使用語言模型來預測下一個詞的概率`成`使用 語言 模型 來 預測 下 一個 詞 的 概率`
- 使用 BERT Tokenizer (WordPiece tokenizer)將每個word分成更小的subwords，在中文中會將每個字視作獨立單位
### Answer Span

1. How did you convert the answer span start/end position on characters to position on tokens after BERT tokenization? 
	- 在tokenizer 中將return_offsets_mapping=True設為true，就會回傳offset_mapping，提供每個 token 在原始上下文中的位置映射。
	- 如`[CLS], The, quick, brown, fox, jumps, over, the, lazy, dog, [SEP]`
	-  offset_mapping 會告訴我們該 token 在原始上下文中的字元範圍：'The' 對應字元範圍 (0, 3), 'quick' 對應字元範圍 (4, 9)...
		```
		tokenized_examples = tokenizer(
				examples[question_column_name if pad_on_right else context_column_name],
				examples[context_column_name if pad_on_right else question_column_name],
				truncation="only_second" if pad_on_right else "only_first",
				max_length=max_seq_length,
				stride=args.doc_stride,
				return_overflowing_tokens=True,
				return_offsets_mapping=True,
				padding="max_length" if args.pad_to_max_length else False,
			)
		```
	-  假設答案是 "jumps over"，它在上下文中的字元範圍是 (20, 30)，接下來會將答案的字元範圍 (20, 30) 轉換為 token 範圍。
		- start_char = 20
		- end_char = 30
	- 找到答案段落的起始 token，和段落的end token
		```
		token_start_index = 0
		while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
		token_start_index += 1
		```
		```
		token_end_index = len(input_ids) - 1
		while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
		token_end_index -= 1

		```
	-  程式會遍歷 token，直到找到一個 token 其起始字元不再小於 start_char（20），此時將 token_start_index - 1 設為答案的起始位置。
		```
		else:
		while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
			token_start_index += 1
		tokenized_examples["start_positions"].append(token_start_index - 1)
		```
	- 以類似方式找到end token位置

		```
		while offsets[token_end_index][1] >= end_char:
		token_end_index -= 1
		tokenized_examples["end_positions"].append(token_end_index + 1)
		```
	-  起始 token 是 'jumps'，其 token_start_index 將是 5。結束 token 'over'，其 token_end_index 為 6。
	- 將 token 索引添加到訓練資料中
		```
		tokenized_examples["start_positions"] = []
		tokenized_examples["end_positions"] = []
		```
2. After your model predicts the probability of answer span start/end position, what rules did you apply to determine the final start/end position?
	- 寫在這個function裡面`def postprocess_qa_predictions`
		- 計算得分最高的起始和結束位置，只會考慮前 n_best_size (default 20) 個最可能的 start 和 end logits 的 token 位置
		```
		start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
		end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

		```
		-  過濾不合理的範圍
			-  檢查答案長度是否超過 max_answer_length，或結束位置是否早於起始位置
			```
			if end_index < start_index or end_index - start_index + 1 > max_answer_length:
            continue
			```
			-  如果上述條件都滿足，則該範圍會被加入到 prelim_predictions 中，並保存該範圍的 start logits 和 end logits 得分加總。
			```
			prelim_predictions.append(
            {
                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                "score": start_logits[start_index] + end_logits[end_index],
                "start_logit": start_logits[start_index],
                "end_logit": end_logits[end_index],
            }

			```
		- 對 prelim_predictions 根據'score'進行排序，只保留 n_best_size 個最高的預測 `predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]`
		- 最後選擇score最高的組合作為最終預測。`all_predictions[example["id"]] = predictions[0]["text"]`



## Q2: Modeling with BERTs and their variants 
### describe
#### Your model.
我在multiple choice 中使用hfl/chinese-bert-wwm-ext，QA中使用hfl/chinese-macbert-large，這兩個model是同一個作者出的，chinese-bert-wwm-ext
- BERT-wwm-ext 保持了與 BERT 相同的架構， 12 層的 Transformer 編碼器，每層有 768 個隱藏單元，參數總量約 1.1 億。
	- pretraining時使用了whole word masking
- chinese-macbert-large使用了MLM as Correction，，以減少預訓練和微調階段之間的差異。
	- MacBERT-large 則是基於 BERT-large 結構，具有 24層Transformer編碼器，每層的隱藏單元大小為 1024，總參數量約為 3.4億。
	- MacBERT並不使用[MASK]標記遮蔽詞彙，而是使用與原詞相似的詞來替代。
		- 其中15% input會被mask，80% 替換為相似詞(word2vec計算)，10% 替換為隨機詞，10% 保持原樣
	- 除了使用到使用N-gram遮蔽策略，即同時替換多個連續的詞，這個模型用到4-gram
- 這兩個模型都使用到ext的資料作訓練，除了中文維基百科的4B資料外，蒐集了更多資料來源如百科全書、新聞、問答網站，擴展到5.4B，內容包含繁體中文和簡體中文。
#### The performance of your model
multiple choice的validation accuracy為0.9687，QA validation set exact match的值為83%，整個model的predict結果丟到kaggle，只有0.77
#### The loss function you used.
1. multiple choice
模型是使用 AutoModelForQuestionAnswering 類別來載入的，根據所選的pretraining model選擇具體的模型類別載入，如BertForMultipleChoice。
定義在`transformers/src/transformers/models/bert
/modeling_bert.py class BertForMultipleChoice(BertPreTrainedModel):`
loss function為CrossEntropyLoss
```
if labels is not None:
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(reshaped_logits, labels)
```
2. QA
模型是使用 AutoModelForQuestionAnswering 類別來載入的，根據所選的pretraining model選擇具體的模型類別載入，如BertForQuestionAnswering。
定義在`transformers/src/transformers/models/bert
/modeling_bert.py class BertForQuestionAnswering(BertPreTrainedModel):`
loss function為CrossEntropyLoss，total_loss 是開始位置損失 start_loss 和結束位置損失 end_loss 的平均值。
```
loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
start_loss = loss_fct(start_logits, start_positions)
end_loss = loss_fct(end_logits, end_positions)
total_loss = (start_loss + end_loss) / 2
```
CrossEntropyLoss主要用於分類問題。結合了 Softmax 和 Negative Log Likelihood Loss (NLLLoss)
- Softmax：將模型的輸出轉換為機率分佈
- NLLLoss：計算真實標籤的預測概率的負對數，概率越接近 1，損失越小；概率越小，損失越大。

### The optimization algorithm (e.g. Adam), learning rate and batch size.
both model
- optimization algorithm: AdamW optimizer,為Adam 的改進，引進了weight decay避免overfitting，我將weight decay設為0.01和0.02。
- Adam 是一種基於一階矩估計的優化器，它結合了 Momentum 和 RMSProp 兩種技術。
- learning rate: 2e-5, scheduler type: linear 
- --per_device_train_batch_size 4(multiple choice), 8(QA)
- --gradient_accumulation_steps 4
- effective batch size: 16 
### Try another type of pre-trained LMs and describe
#### The new model
bert-base-chinese
#### The performance of this model
multiple choice的validation accuracy降到0.9584，QA validation set exact match的值降到79%，整個model的predict結果丟到kaggle，只有0.73
#### The difference between pre-trained LMs (architecture, pretraining loss, etc.)
- BERT 的掩碼策略是基於「詞片段（subword）」進行隨機遮罩，即它會隨機遮罩文本中的子詞（subword），無論它是否為一個完整的單詞。
- 訓練數據為中文 Wikipedia，較少(1.7B)
- 模型架構：與 BERT-wwm-ext 相同
- 模型大小：Base (12-layer, 768-hidden, 12-heads, 110M parameters)

## Q3: Curves
![W&B Chart 2024_10_5 上午1_24_28](https://hackmd.io/_uploads/SJ31Sj6RA.png)
![W&B Chart 2024_10_5 上午1_22_56 (1)](https://hackmd.io/_uploads/By21HipCA.png)

## Q4: Pre-trained vs Not Pre-trained 
我是將multiple choice 問題用not pre-trained 在訓練一次，tokenizer用bert-base-uncased ，以下是training hyperparameters
```
python ./from_scratch.py \
  --model_type bert \
  --tokenizer_name bert-base-uncased \
  --max_seq_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --lr_scheduler_type linear \
  --train_file ./data/multiplechoice_train_data.json \
  --validation_file ./data/multiplechoice_validation_data.json \
  --with_tracking \
  --report_to wandb \
  --output_dir ./from_scratch \
  --checkpointing_steps epoch
```
在validation上的performance，最後只有0.52跟pretrain model差很多
![W&B Chart 2024_10_7 下午4_14_21](https://hackmd.io/_uploads/Sk5_OzbJyx.png)
loss也降不太下來
![W&B Chart 2024_10_7 下午4_15_43](https://hackmd.io/_uploads/S1fWiMW1kx.png)
因此使用pretrainded model是較好的選擇