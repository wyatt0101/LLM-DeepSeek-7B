from modelscope import snapshot_download

model_dir = snapshot_download(
    'deepseek-ai/deepseek-llm-7b-chat',
    cache_dir=r'E:/phd_documents/self-llm/models/DeepSeek/models',
    revision='master'
)

print("模型下载完成，路径为：", model_dir)