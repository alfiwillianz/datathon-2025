#!/usr/bin/env python3
"""
Batch merge script for all SemiQwenn student models
Merges trained LoRA adapters with their base models
"""

import os
import sys
import json
import time
import logging
import torch
import gc
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages:")
    print("  pip install transformers peft torch")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_model_configs():
    """Get all model configurations for merging"""
    return [
        {
            "name": "qwen2.5-0.5b",
            "base_model": "StudentModels/qwen_models/2.5-0.5B/",
            "adapter": "StudentModels/Adapters/qwen2.5-0.5b-lora/checkpoint-4122/",
            "output": "StudentModels/MergedModels/qwen2.5-0.5b-merged/"
        },
        {
            "name": "qwen2.5-1.5b", 
            "base_model": "StudentModels/qwen_models/2.5-1.5B/",
            "adapter": "StudentModels/Adapters/qwen2.5-1.5b-lora/",  # Use root dir with final adapter
            "output": "StudentModels/MergedModels/qwen2.5-1.5b-merged/"
        },
        {
            "name": "qwen2.5-7b",
            "base_model": "StudentModels/qwen_models/2.5-7B/",
            "adapter": "StudentModels/Adapters/qwen2.5-7b-qlora/checkpoint-4122/",
            "output": "StudentModels/MergedModels/qwen2.5-7b-merged/"
        }
    ]

def validate_paths(config):
    """Validate that all required paths exist"""
    base_exists = os.path.exists(config["base_model"])
    adapter_exists = os.path.exists(config["adapter"])
    
    if not base_exists:
        logger.error(f"Base model not found: {config['base_model']}")
    if not adapter_exists:
        logger.error(f"Adapter not found: {config['adapter']}")
    
    return base_exists and adapter_exists

def merge_single_model(config):
    """Merge a single model configuration"""
    logger.info(f"🚀 Starting merge: {config['name']}")
    logger.info(f"📁 Base: {config['base_model']}")
    logger.info(f"📁 Adapter: {config['adapter']}")
    logger.info(f"📁 Output: {config['output']}")
    
    start_time = time.time()
    
    try:
        # Create output directory
        os.makedirs(config["output"], exist_ok=True)
        
        # Load tokenizer
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"],
            trust_remote_code=True
        )
        
        # Load base model
        logger.info("📥 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load PEFT adapter
        logger.info("📥 Loading PEFT adapter...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            config["adapter"],
            torch_dtype=torch.float16
        )
        
        # Merge models
        logger.info("🔄 Merging models...")
        merged_model = peft_model.merge_and_unload()
        
        # Save merged model
        logger.info("💾 Saving merged model...")
        merged_model.save_pretrained(
            config["output"],
            safe_serialization=True,
            max_shard_size="2GB"
        )
        
        # Save tokenizer
        logger.info("💾 Saving tokenizer...")
        tokenizer.save_pretrained(config["output"])
        
        # Save merge info
        merge_info = {
            "model_name": config["name"],
            "base_model": config["base_model"],
            "adapter": config["adapter"],
            "merge_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch_dtype": "float16",
            "merged_by": "SemiQwenn Batch Merger"
        }
        
        with open(os.path.join(config["output"], "merge_info.json"), "w") as f:
            json.dump(merge_info, f, indent=2)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ {config['name']} merged successfully in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to merge {config['name']}: {str(e)}")
        return False
        
    finally:
        # Cleanup
        for var in ['merged_model', 'peft_model', 'base_model', 'tokenizer']:
            if var in locals():
                del locals()[var]
        torch.cuda.empty_cache()
        gc.collect()

def main():
    """Main function to merge all models"""
    logger.info("🤖 SemiQwenn Batch Model Merger")
    logger.info("=" * 50)
    
    # Get configurations
    configs = get_model_configs()
    
    # Validate all paths first
    logger.info("🔍 Validating model paths...")
    valid_configs = []
    for config in configs:
        if validate_paths(config):
            valid_configs.append(config)
            logger.info(f"✅ {config['name']}: Ready for merge")
        else:
            logger.warning(f"❌ {config['name']}: Missing files, skipping")
    
    if not valid_configs:
        logger.error("❌ No valid model configurations found!")
        return False
    
    logger.info(f"📋 Found {len(valid_configs)} valid models to merge")
    
    # Create base output directory
    os.makedirs("StudentModels/MergedModels", exist_ok=True)
    
    # Merge each model
    results = {}
    total_start = time.time()
    
    for i, config in enumerate(valid_configs, 1):
        logger.info("=" * 60)
        logger.info(f"🎯 Merging {i}/{len(valid_configs)}: {config['name']}")
        logger.info("=" * 60)
        
        success = merge_single_model(config)
        results[config['name']] = success
        
        # Brief pause between merges
        if i < len(valid_configs):
            time.sleep(2)
    
    # Summary
    total_time = time.time() - total_start
    successful = sum(results.values())
    total_models = len(results)
    
    logger.info("=" * 60)
    logger.info("📊 MERGE SUMMARY")
    logger.info("=" * 60)
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{status}: {model_name}")
    
    logger.info(f"🎯 Results: {successful}/{total_models} models merged")
    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    
    if successful == total_models:
        logger.info("🎉 All models merged successfully!")
        logger.info("📁 Merged models available in: StudentModels/MergedModels/")
    else:
        logger.warning(f"⚠️  {total_models - successful} models failed")
    
    return successful == total_models

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
