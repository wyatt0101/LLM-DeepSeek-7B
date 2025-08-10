from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class DeepSeek_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.bfloat16,  device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        messages = [
            {"role": "user", "content": prompt}
        ]
        # 构建输入     
        input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        # 通过模型获得输出
        outputs = self.model.generate(input_tensor.to(self.model.device), max_new_tokens=1000)
        response = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return response
        
    @property
    def _llm_type(self) -> str:
        return "DeepSeek_LLM"