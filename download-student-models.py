from huggingface_hub import hf_hub_download
import os

def download_model(repo_id, filename, cache_dir=None):
    try:
        print(f" Downloading {filename} from {repo_id}...")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        print(f"Downloaded to: {model_path}\n")
        return model_path
    except Exception as e:
        print(f"Failed to download {filename} from {repo_id}: {e}")
        return None

def download_student_models():
    # === Qwen 1.5B ===
    qwen_files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ]

    for file in qwen_files:
        download_model(
            repo_id="Qwen/Qwen2.5-1.5B",
            filename=file,
            cache_dir="./StudentModels/Qwen"
        )

    # === Qwen 0.5B ===
    qwen_05b_files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ]

    for file in qwen_05b_files:
        download_model(
            repo_id="Qwen/Qwen2.5-0.5B",
            filename=file,
            cache_dir="./StudentModels/Qwen0.5B"
        )

    # === Mistral 7B Instruct ===
    mistral_files = [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.model",
        "tokenizer_config.json"
    ]

    for file in mistral_files:
        download_model(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            filename=file,
            cache_dir="./StudentModels/Mistral"
        )

if __name__ == "__main__":
    download_student_models()
