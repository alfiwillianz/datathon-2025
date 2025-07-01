# dataset-prep.py
from datasets import load_dataset
import random
import json

def format_dataset(dataset, input_key, output_key, max_samples=None):
    samples = []
    for item in dataset:
        prompt = item.get(input_key, "").strip()
        answer = item.get(output_key, "").strip()
        if prompt and answer:
            samples.append({"instruction": prompt, "output": answer})
    if max_samples:
        samples = random.sample(samples, min(max_samples, len(samples)))
    return samples

# Load CodeAlpaca (~20k)
codealpaca = format_dataset(
    load_dataset("sahil2801/CodeAlpaca-20k", split="train"),
    "instruction", "output", 20000
)

# Load GSM8K (OpenAI) (~8.5k), use 'question' + 'answer'
gsm = format_dataset(
    load_dataset("openai/gsm8k", "main", split="train"),
    "question", "answer", 10000
)

# Combine and shuffle
combined = codealpaca + gsm
random.shuffle(combined)

# Save as JSONL
with open("Dataset/codealpaca_gsm8k_30k.jsonl", "w") as f:
    for sample in combined:
        f.write(json.dumps(sample) + "\n")

print("Saved as codealpaca_gsm8k_30k.jsonl")
