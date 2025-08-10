import os
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch
from torch.utils.data import Dataset


# 1. 路径和参数
model_path = r"./models/deepseek-ai/deepseek-llm-7b-chat"
output_dir = "./output/DeepSeek_LoRA"
max_seq_length = 384
batch_size = 16
num_train_epochs = 3
learning_rate = 1e-4


# 2. 加载 tokenizer 和基础模型
print("加载 tokenizer 和模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.generation_config = GenerationConfig.from_pretrained(model_path)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
# model.eval()


# 3. 配置 LoRA
print("配置 LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 显示可训练参数数量

model.gradient_checkpointing_enable()
model.enable_input_require_grads()


# 4. 准备训练数据
json_path = "./huanhuan.json"
with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)  # raw_data是list，每项是 dict



# 5. 数据预处理函数
def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(f"User: {example['instruction']+example['input']}\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"Assistant: {example['output']}<｜end▁of▁sentence｜>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]

    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1

    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 预处理训练数据
print("预处理训练数据...")

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }
    
# 把预处理好的列表传给 Dataset
tokenized_dataset = [process_func(x) for x in raw_data]
train_dataset = MyDataset(tokenized_dataset)


# 6. 定义训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    num_train_epochs=num_train_epochs,
    logging_steps=10,
    save_steps=50,
    learning_rate=learning_rate,
    save_total_limit=2,
    fp16=True,
    # evaluation_strategy="no",
    save_strategy="steps",
    gradient_checkpointing=True,
    report_to=[]
)


# 7. 创建 Trainer 并开始训练
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

print("开始训练...")
trainer.train()


# 8. 微调完成后推理测试
print("测试微调后的模型...")
test_text = "小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——"
inputs = tokenizer(f"User: {test_text}\n\n", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("模型回答：", result)

