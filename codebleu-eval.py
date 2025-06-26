import json
from nltk.translate.bleu_score import corpus_bleu

# File path to your JSONL
jsonl_path = "./Dataset/categorized_split_data_stratified_test.jsonl"

# Collect predictions and references
references = []
predictions = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("category") != "codealpaca":
            continue

        ref = obj.get("expected_output", "").strip()
        pred = obj.get("model_output", "").strip()

        if not ref or not pred:
            continue

        # Append tokenized versions
        references.append([ref.split()])
        predictions.append(pred.split())

# Check if we have valid samples
total = len(references)
if total == 0:
    print("❌ No valid codealpaca samples found.")
else:
    # Compute BLEU
    bleu_score = corpus_bleu(references, predictions)
    print(f"\n📊 [CodeAlpaca] BLEU Score: {bleu_score:.4f} on {total} samples")
