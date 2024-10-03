from transformers import pipeline

# 指定 device 為 0，這樣模型會在 GPU 上運行
classifier = pipeline('sentiment-analysis', device=0)

# 測試情感分析
result = classifier("This is a fantastic example!")
print(result)
