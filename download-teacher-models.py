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

def download_teacher_models():
    # === DevStral GGUF ===
    download_model(
        repo_id="mistralai/Devstral-Small-2505_gguf",
        filename="devstralQ4_K_M.gguf",
        cache_dir="./Model"
    )

if __name__ == "__main__":
    download_teacher_models()
