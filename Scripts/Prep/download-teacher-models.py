from huggingface_hub import hf_hub_download
from pathlib import Path
import os

def download_model(repo_id, filename, cache_dir=None):
    """
    Download a single model file from Hugging Face Hub.
    
    Args:
        repo_id (str): Repository ID on Hugging Face Hub
        filename (str): Name of the file to download
        cache_dir (str, optional): Directory to cache the model
    
    Returns:
        str: Path to downloaded model, or None if failed
    """
    try:
        print(f"📥 Downloading {filename} from {repo_id}...")
        
        # Ensure cache directory exists
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        print(f"✓ Downloaded successfully to: {model_path}\n")
        return model_path
        
    except Exception as e:
        print(f"✗ Failed to download {filename} from {repo_id}: {e}\n")
        return None

def download_teacher_models():
    """Download all teacher models required for the project."""
    print("Starting teacher models download...\n")
    
    # Define models to download
    models_config = [
        {
            "repo_id": "mistralai/Devstral-Small-2505_gguf",
            "filename": "devstralQ4_K_M.gguf",
            "cache_dir": "Model"
        }
    ]
    
    # Track download results
    success_count = 0
    total_count = len(models_config)
    
    # Download each model
    for config in models_config:
        result = download_model(**config)
        if result:
            success_count += 1
    
    # Summary
    print("=" * 50)
    print(f"Download Summary: {success_count}/{total_count} models downloaded successfully")
    
    if success_count == total_count:
        print("🎉 All teacher models downloaded successfully!")
        return True
    else:
        print("⚠️  Some downloads failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    download_teacher_models()
