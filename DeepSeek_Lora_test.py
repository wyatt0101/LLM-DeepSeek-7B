from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch

def main():
    # 基础模型路径（没改的话就是原始模型路径）
    base_model_path = r"./models/deepseek-ai/deepseek-llm-7b-chat"
    # LoRA微调权重保存路径
    lora_model_path = r"./output/DeepSeek_LoRA_zhenhuan/checkpoint-351"

    print("加载 tokenizer 和基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.generation_config = GenerationConfig.from_pretrained(base_model_path)
    base_model.generation_config.pad_token_id = base_model.generation_config.eos_token_id

    print("加载微调的 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # 生成函数
    def generate_response(text, max_new_tokens=100):
        inputs = tokenizer(f"User: {text}\n\n", return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    # 测试推理
    prompt = "这个温太医啊，也是古怪，谁不知太医不得皇命不能为皇族以外的人请脉诊病，他倒好，十天半月便往咱们府里跑。"
    print("输入：", prompt)
    answer = generate_response(prompt)
    print("模型回答：", answer)

if __name__ == "__main__":
    main()
