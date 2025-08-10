# import requests

# url = "http://127.0.0.1:6006/"
# data = {
#     "prompt": "怎么在一个月内减30斤",
#     "max_length": 1000
# }

# response = requests.post(url, json=data)
# print(response.json())


# # import torch

# # print(torch.__version__)            # PyTorch版本
# # print(torch.version.cuda)           # CUDA版本（应该是12.1）
# # print(torch.cuda.is_available())    # True 才说明可以用
# # print(torch.cuda.get_device_name(0))  # 显卡型号


# langchain多轮对话
from DeepSeek_LLM import DeepSeek_LLM


mode_name_or_path = './models/deepseek-ai/deepseek-llm-7b-chat'
llm = DeepSeek_LLM(mode_name_or_path)

response = llm.invoke('请你告诉我如何本地部署大模型')
print(response)