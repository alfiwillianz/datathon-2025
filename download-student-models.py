from huggingface_hub import snapshot_download
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

path = "./StudentModels/"

# ==============================
# Qwen2.5-7B
# ==============================
qwen_7b_path = Path(path).joinpath('qwen_models', '2.5-7B')
qwen_7b_path.mkdir(parents=True, exist_ok=True)

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
    local_dir=qwen_7b_path,
    token=hf_token
)

# ==============================
# Qwen2.5-1.5B
# ==============================
qwen_15b_path = Path(path).joinpath('qwen_models', '2.5-1.5B')
qwen_15b_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B",
    allow_patterns=[
        "model.safetensors", "config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt"
    ],
    local_dir=qwen_15b_path,
    token=hf_token
)

# ==============================
# Qwen2.5-0.5B
# ==============================
qwen_05b_path = Path(path).joinpath('qwen_models', '2.5-0.5B')
qwen_05b_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B",
    allow_patterns=[
        "model.safetensors", "config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt"
    ],
    local_dir=qwen_05b_path,
    token=hf_token
)
