from llama_cpp import Llama
import json
from tqdm import tqdm
import time
import os

# ========== Environment Setup ==========
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ========== Load GGUF Model to GPU Only ==========
llm = Llama(
    model_path="Model/devstralQ4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    n_threads=os.cpu_count(),
    use_mlock=True,
    n_batch=1024,
    use_mmap=False,
    verbose=False,
    seed=42
)
print("Model loaded fully to GPU", flush=True)

# ========== Load Dataset ==========
with open("Dataset/Prep/codealpaca_gsm8k_30k.jsonl") as f:
    all_data = [json.loads(line) for line in f]

# ========== Resume Support ==========
partial_file = "Dataset/Partial/devstral_inference_partial.jsonl"  # Updated path
already_done = 0

if os.path.exists(partial_file):
    with open(partial_file) as f:
        already_done = sum(1 for _ in f)

print(f"Resuming from sample {already_done}", flush=True)
dataset = all_data[already_done:]

print(f"Loaded {len(dataset)} samples to process", flush=True)

# ========== Inference Loop ==========
start_time = time.time()
batch_size = 32
backup_interval = 5  # Save full backup every 5 batches

with tqdm(total=len(dataset), desc="Inference Progress", unit="samples") as pbar:
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batch_results = []

        for item in batch:
            instruction = str(item.get("instruction", ""))[:4000]
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

            try:
                response = llm(
                    prompt,
                    max_tokens=256,
                    stop=["###", "</s>"],
                    temperature=0.1,
                    repeat_penalty=1.1,
                    top_p=0.9,
                    top_k=40
                )
                output = response.get("choices", [{}])[0].get("text", "").strip()
            except Exception as e:
                output = f"ERROR: {str(e)[:100]}"

            batch_results.append({
                "instruction": instruction,
                "model_output": output
            })

        # Save every batch with failsafe
        try:
            with open(partial_file, "a") as f:
                for r in batch_results:
                    f.write(json.dumps(r) + "\n")
        except Exception as save_error:
            backup_file = f"./Dataset/Backup/devstral_inference_backup_{int(time.time())}.jsonl"
            try:
                with open(backup_file, "w") as bf:
                    for r in batch_results:
                        bf.write(json.dumps(r) + "\n")
                print(f"Error saving to main file: {save_error}. Backup saved to {backup_file}", flush=True)
            except Exception as backup_error:
                print(f"CRITICAL: Failed to save batch results: {backup_error}", flush=True)

        # Periodic backup
        if (i // batch_size) % backup_interval == 0 and i > 0:
            backup_file = f"./Dataset/Backup/evstral_inference_backup_{i}.jsonl"
            try:
                with open(backup_file, "w") as f:
                    with open(partial_file, "r") as source:
                        f.write(source.read())
                print(f"Periodic backup saved: {backup_file}", flush=True)
            except Exception as e:
                print(f"Failed to create periodic backup: {e}", flush=True)

        pbar.update(len(batch_results))

        # ETA display
        elapsed = time.time() - start_time
        processed = i + len(batch_results)
        avg_time = elapsed / processed if processed > 0 else 0
        remaining = avg_time * (len(dataset) - processed)
        pbar.set_postfix({
            'avg': f"{avg_time:.2f}s/sample",
            'ETA': f"{remaining / 60:.1f} min"
        })

# ========== Completion ==========
elapsed = time.time() - start_time
print(f"\nInference completed in {elapsed / 60:.2f} minutes", flush=True)
print(f"Average time per sample: {elapsed / len(dataset):.2f} seconds", flush=True)
print(f"Results saved to: {partial_file}", flush=True)
