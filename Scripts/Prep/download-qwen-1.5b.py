from huggingface_hub import snapshot_download
from pathlib import Path
from dotenv import load_dotenv
import os

def download_qwen_1_5b():
    """Download Qwen2.5-1.5B model files."""
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Warning: HF_TOKEN not found. This may cause issues with private repositories.")
    
    base_path = Path("StudentModels/")
    model_path = base_path.joinpath('qwen_models', '2.5-1.5B')
    model_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Downloading Qwen2.5-1.5B model...")
        snapshot_download(
            repo_id="Qwen/Qwen2.5-1.5B",
            allow_patterns=[
                "model.safetensors", 
                "config.json", 
                "tokenizer.json",
                "tokenizer_config.json", 
                "vocab.json", 
                "merges.txt"
            ],
            local_dir=model_path,
            token=hf_token
        )
        print(f"✓ Qwen2.5-1.5B downloaded successfully to: {model_path}")
        
    except Exception as e:
        print(f"✗ Failed to download Qwen2.5-1.5B: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_qwen_1_5b()
