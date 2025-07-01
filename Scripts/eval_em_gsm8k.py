import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

def extract_final_answer(text):
    """Extract final answer after '####'."""
    return text.strip().split("####")[-1].strip() if "####" in text else None

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_gsm8k_lookup(devstral_data):
    return {
        d["instruction"]: extract_final_answer(d["expected_output"])
        for d in devstral_data
        if d.get("category") == "gsm8k"
    }

def evaluate_model_outputs(model_data, instr_to_expected, model_name):
    results = []
    for entry in model_data:
        instr = entry["instruction"]
        predicted = extract_final_answer(entry["output"])
        if instr in instr_to_expected:
            expected = instr_to_expected[instr]
            is_correct = int(predicted == expected)
            results.append({
                "model": model_name,
                "instruction": instr,
                "expected": expected,
                "predicted": predicted,
                "em": is_correct
            })
    return results

# Paths
devstral_path = Path("Dataset/Inference/devstral.jsonl")

# Load GSM8K ground-truth
devstral_data = load_jsonl(devstral_path)
instr_to_expected = build_gsm8k_lookup(devstral_data)

# Model file registry
model_files = {
    "qwen-base-0.5b": "Dataset/Inference/Base/qwen-base-2.5-0.5b_base_output.jsonl",
    "qwen-base-1.5b": "Dataset/Inference/Base/qwen-base-2.5-1.5b_base_output.jsonl",
    "qwen-base-7b":   "Dataset/Inference/Base/qwen-base-2.5-7b_base_output.jsonl",
    "semiqwen-0.5b":  "Dataset/Inference/student/qwen2.5-0.5b_output.jsonl",
    "semiqwen-1.5b":  "Dataset/Inference/student/qwen2.5-1.5b_output.jsonl",
    "semiqwen-7b":    "Dataset/Inference/student/qwen2.5-7b_output.jsonl"
}

# Evaluate
all_results = []
for model_name, path_str in model_files.items():
    path = Path(path_str)
    if path.exists():
        data = load_jsonl(path)
        results = evaluate_model_outputs(data, instr_to_expected, model_name)
        all_results.extend(results)
    else:
        print(f"⚠️  Missing file: {path}")

# Summary
summary = defaultdict(lambda: {"correct": 0, "total": 0})
for r in all_results:
    summary[r["model"]]["correct"] += r["em"]
    summary[r["model"]]["total"] += 1

# Print EM accuracy per model
print("\n📊 GSM8K Exact Match Accuracy")
for model, stats in summary.items():
    acc = stats["correct"] / stats["total"] if stats["total"] else 0
    print(f"{model:<15} EM: {acc:.2%} ({stats['correct']}/{stats['total']})")

# Save detailed results
df = pd.DataFrame(all_results)
df.to_csv("gsm8k_em_results.csv", index=False)
print("\n✅ Results saved to gsm8k_em_results.csv")
