from huggingface_hub import snapshot_download
from pathlib import Path
from dotenv import load_dotenv
import os

def download_qwen_7b():
    """Download Qwen2.5-7B model files."""
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Warning: HF_TOKEN not found. This may cause issues with private repositories.")
    
    base_path = Path("StudentModels/")
    model_path = base_path.joinpath('qwen_models', '2.5-7B')
    model_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Downloading Qwen2.5-7B model...")
        snapshot_download(
            repo_id="Qwen/Qwen2.5-7B",
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "merges.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "model.safetensors.index.json",
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors"
            ],
            local_dir=model_path,
            token=hf_token
        )
        print(f"✓ Qwen2.5-7B downloaded successfully to: {model_path}")
        
    except Exception as e:
        print(f"✗ Failed to download Qwen2.5-7B: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_qwen_7b()
