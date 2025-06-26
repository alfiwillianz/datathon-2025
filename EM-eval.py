import json
import re

def extract_gsm8k_answer(text):
    # Try to find answer after '####'
    match = re.search(r"####\s*(\d+)", text)
    if match:
        return match.group(1).strip()
    
    # Fallback: take last number in the text
    numbers = re.findall(r"\b\d+\b", text)
    return numbers[-1] if numbers else None

def normalize(text):
    return text.strip().lower()

# File path
file_path = "./Dataset/categorized_split_data_stratified_test.jsonl"

# Counters
total = 0
correct = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        if obj.get("category", "").lower() != "gsm8k":
            continue  # skip non-GSM8K

        pred_ans = extract_gsm8k_answer(obj.get("model_output", ""))
        gold_ans = extract_gsm8k_answer(obj.get("expected_output", ""))

        if pred_ans is None or gold_ans is None:
            continue  # skip if can't extract answer

        total += 1
        if normalize(pred_ans) == normalize(gold_ans):
            correct += 1

# Result
em = (correct / total) * 100 if total else 0
print(f"[GSM8K] Exact Match (EM): {em:.2f}% ({correct} / {total})")
