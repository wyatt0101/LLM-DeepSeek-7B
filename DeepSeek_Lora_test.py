# 导入必要的库
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig  # Hugging Face Transformers库
from peft import PeftModel  # PEFT库，用于加载LoRA微调权重
import torch  # PyTorch库

def main():
    # 基础模型路径（如果没改，就是原始的预训练模型路径）
    base_model_path = r"./models/deepseek-ai/deepseek-llm-7b-chat"
    # LoRA微调权重保存路径（checkpoint文件夹）
    lora_model_path = r"./output/DeepSeek_LoRA_zhenhuan/checkpoint-351"

    print("加载 tokenizer 和基础模型...")
    # 加载 tokenizer（负责将文本转为token id）
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"  # 设置padding方向为右侧

    # 加载基础模型（Causal LM，用于生成任务）
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # 加载生成配置，例如eos_token_id、pad_token_id等
    base_model.generation_config = GenerationConfig.from_pretrained(base_model_path)
    # 将pad_token_id设置为eos_token_id，避免生成过程中padding报错
    base_model.generation_config.pad_token_id = base_model.generation_config.eos_token_id

    print("加载微调的 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)  # 将LoRA权重加载到基础模型
    model.eval()  # 切换模型为推理模式，不计算梯度
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # 移动模型到GPU或CPU


    def generate_response(text, max_new_tokens=100):
        # 使用tokenizer将文本转换为模型输入张量，并移动到模型设备
        inputs = tokenizer(f"User: {text}\n\n", return_tensors="pt").to(model.device)
        # 调用模型生成文本
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        # 解码生成的token为可读文本
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    prompt = "这个温太医啊，也是古怪，谁不知太医不得皇命不能为皇族以外的人请脉诊病，他倒好，十天半月便往咱们府里跑。"
    print("输入：", prompt)
    answer = generate_response(prompt)
    print("模型回答：", answer)

if __name__ == "__main__":
    main()
