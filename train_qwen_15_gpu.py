from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
import psutil
import logging
import os
import subprocess

# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
process = psutil.Process(os.getpid())

def log_resource_usage():
    cpu = process.cpu_percent(interval=None)
    mem = process.memory_info().rss / (1024 * 1024)
    logging.info(f"CPU: {cpu:.1f}% | Mem: {mem:.2f} MB")

def log_gpu_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        gpu_stats = result.stdout.strip().split('\n')
        for idx, line in enumerate(gpu_stats):
            util, mem_used, mem_total = map(int, line.split(', '))
            logging.info(f"GPU {idx} - Util: {util}% | Mem: {mem_used} MiB / {mem_total} MiB")
    except Exception as e:
        logging.warning(f"GPU logging failed: {e}")

# =========================
# Load Dataset
# =========================
dataset = load_dataset("json", data_files={"train": "./Dataset/categorized_split_data_stratified_train.jsonl"})["train"]
dataset = dataset.map(lambda x: {"text": x["instruction"] + "\n" + x["model_output"]})

# =========================
# Tokenizer
# =========================
model_dir = "./StudentModels/mistral_models/7B-Instruct-v0.3/"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)  # Set use_fast=False if you get tiktoken error
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# =========================
# Model + LoRA Config
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# =========================
# Training Arguments
# =========================
training_args = TrainingArguments(
    output_dir="./mistral-qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

# =========================
# Resource Monitor Callback
# =========================
losses = []

class ResourceMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            loss_val = logs["loss"]
            losses.append(loss_val)
            logging.info(f"Step {state.global_step} | Loss: {loss_val:.4f}")
        log_resource_usage()
        log_gpu_usage()

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[ResourceMonitorCallback()]
)

# =========================
# Start Training
# =========================
trainer.train()

# =========================
# Plot Loss Curve
# =========================
plt.plot(losses, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Mistral 7B LoRA Training Loss')
plt.legend()
plt.tight_layout()
plt.savefig('mistral_7b_training_loss.png')
print("📊 Saved loss plot to mistral_7b_training_loss.png")
