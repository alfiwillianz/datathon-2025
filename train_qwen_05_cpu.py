import os
import psutil
import torch
import logging
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader

# -------------------------------
# Configuration
# -------------------------------
model_name = "./StudentModels/qwen_models/2.5-0.5B/"
data_path = "./Dataset/categorized_split_data_stratified_train.jsonl"
cpu_threads = 4  # out of 12 threads on Ryzen 5 7500F
batch_size = 2
epochs = 3

# -------------------------------
# Thread Limiting
# -------------------------------
os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
torch.set_num_threads(cpu_threads)
print(f"🧠 Using {cpu_threads} CPU threads for training")

# -------------------------------
# Logger Setup
# -------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
process = psutil.Process(os.getpid())

def log_resource_usage(step=None):
    cpu = process.cpu_percent(interval=None)
    mem = process.memory_info().rss / (1024 * 1024)
    msg = f"CPU: {cpu:.1f}% | Mem: {mem:.2f} MB"
    if step is not None:
        msg = f"Step {step} | " + msg
    print(msg)

# -------------------------------
# Load Tokenizer & Model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cpu")
model.to(device)

# -------------------------------
# Load & Tokenize Dataset
# -------------------------------
dataset = load_dataset("json", data_files=data_path)["train"]
dataset = dataset.map(lambda x: {"text": x["instruction"] + "\n" + x["model_output"]})

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# -------------------------------
# Dataloader
# -------------------------------
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=2)

# -------------------------------
# Optimizer
# -------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# -------------------------------
# Training Loop
# -------------------------------
losses = []

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids']
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 10 == 0:
            print(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.4f}")
            log_resource_usage(step)

# -------------------------------
# Plot & Save Loss Graph
# -------------------------------
plt.plot(losses, label="Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Qwen 0.5B CPU Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig("qwen_05_training_loss.png")
print("📉 Training loss plot saved as qwen_05_training_loss.png")
