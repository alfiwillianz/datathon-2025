# Enhanced merge script for all PEFT adapters with base models
import argparse
import os
import torch
import logging
import time
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_merge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def discover_model_pairs():
    """Discover all base model and adapter pairs for merging"""
    model_pairs = []
    
    # Define model configurations
    configs = [
        {
            "name": "qwen2.5-0.5b",
            "base_model": "StudentModels/qwen_models/2.5-0.5B/",
            "adapter_base": "StudentModels/Adapters/qwen2.5-0.5b-lora/",
            "checkpoint": "checkpoint-4122"  # Latest checkpoint
        },
        {
            "name": "qwen2.5-1.5b",
            "base_model": "StudentModels/qwen_models/2.5-1.5B/",
            "adapter_base": "StudentModels/Adapters/qwen2.5-1.5b-lora/",
            "checkpoint": None  # Use root adapter directory (has final adapter files)
        },
        {
            "name": "qwen2.5-7b",
            "base_model": "StudentModels/qwen_models/2.5-7B/",
            "adapter_base": "StudentModels/Adapters/qwen2.5-7b-qlora/",
            "checkpoint": "checkpoint-4122"  # Latest checkpoint
        }
    ]
    
    for config in configs:
        base_path = config["base_model"]
        adapter_path = config["adapter_base"]
        
        # Use specific checkpoint if specified, otherwise use root adapter directory
        if config["checkpoint"]:
            adapter_path = os.path.join(adapter_path, config["checkpoint"])
        
        # Check if paths exist
        if os.path.exists(base_path) and os.path.exists(adapter_path):
            output_path = f"StudentModels/MergedModels/{config['name']}-merged/"
            model_pairs.append({
                "name": config["name"],
                "base_model": base_path,
                "adapter": adapter_path,
                "output": output_path
            })
            logger.info(f"✅ Found model pair: {config['name']}")
        else:
            logger.warning(f"❌ Missing paths for {config['name']}:")
            logger.warning(f"   Base: {base_path} (exists: {os.path.exists(base_path)})")
            logger.warning(f"   Adapter: {adapter_path} (exists: {os.path.exists(adapter_path)})")
    
    return model_pairs

def merge_peft_model(base_model_path, peft_model_path, output_model_path, model_name):
    """Merge PEFT adapter with base model"""
    logger.info(f"🚀 Starting merge for {model_name}")
    logger.info(f"📁 Base model: {base_model_path}")
    logger.info(f"📁 Adapter: {peft_model_path}")
    logger.info(f"📁 Output: {output_model_path}")
    
    start_time = time.time()
    
    try:
        # Create output directory
        os.makedirs(output_model_path, exist_ok=True)
        
        # Load tokenizer first
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # Load base model with optimized settings
        logger.info("📥 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load PEFT adapter
        logger.info("📥 Loading PEFT adapter...")
        peft_model = PeftModel.from_pretrained(
            base_model, 
            peft_model_path,
            torch_dtype=torch.float16
        )
        
        # Merge the models
        logger.info("🔄 Merging models...")
        merged_model = peft_model.merge_and_unload()
        
        # Save merged model
        logger.info("💾 Saving merged model...")
        merged_model.save_pretrained(
            output_model_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        
        # Save tokenizer
        logger.info("💾 Saving tokenizer...")
        tokenizer.save_pretrained(output_model_path)
        
        # Create model info file
        model_info = {
            "model_name": model_name,
            "base_model": base_model_path,
            "adapter": peft_model_path,
            "merge_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch_dtype": "float16",
            "merged_by": "SemiQwenn Model Merger"
        }
        
        import json
        with open(os.path.join(output_model_path, "merge_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ {model_name} merged successfully in {elapsed_time:.2f}s")
        logger.info(f"📁 Saved to: {output_model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to merge {model_name}: {str(e)}")
        return False
        
    finally:
        # Cleanup memory
        if 'merged_model' in locals():
            del merged_model
        if 'peft_model' in locals():
            del peft_model
        if 'base_model' in locals():
            del base_model
        if 'tokenizer' in locals():
            del tokenizer
        
        torch.cuda.empty_cache()
        gc.collect()

def merge_all_models():
    """Merge all discovered model pairs"""
    logger.info("🔍 Discovering available model pairs...")
    model_pairs = discover_model_pairs()
    
    if not model_pairs:
        logger.error("❌ No valid model pairs found!")
        return False
    
    logger.info(f"📋 Found {len(model_pairs)} model pairs to merge")
    
    # Create base output directory
    os.makedirs("StudentModels/MergedModels", exist_ok=True)
    
    results = {}
    total_start_time = time.time()
    
    for i, pair in enumerate(model_pairs, 1):
        logger.info("=" * 60)
        logger.info(f"🎯 Merging {i}/{len(model_pairs)}: {pair['name']}")
        logger.info("=" * 60)
        
        success = merge_peft_model(
            pair['base_model'],
            pair['adapter'],
            pair['output'],
            pair['name']
        )
        
        results[pair['name']] = success
        
        # Brief pause between merges
        if i < len(model_pairs):
            time.sleep(2)
    
    # Summary
    total_time = time.time() - total_start_time
    successful = sum(results.values())
    total_models = len(results)
    
    logger.info("=" * 60)
    logger.info("📊 MODEL MERGE SUMMARY")
    logger.info("=" * 60)
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{status}: {model_name}")
    
    logger.info(f"🎯 Results: {successful}/{total_models} models merged successfully")
    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    
    if successful == total_models:
        logger.info("🎉 All models merged successfully!")
        logger.info("📁 Merged models available in: StudentModels/MergedModels/")
    else:
        logger.warning(f"⚠️  {total_models - successful} models failed to merge")
    
    return successful == total_models

def main():
    parser = argparse.ArgumentParser(description="Merge PEFT adapters with base models")
    parser.add_argument("--base_model_path", type=str, help="Path to the base model")
    parser.add_argument("--peft_model_path", type=str, help="Path to the PEFT model")
    parser.add_argument("--output_model_path", type=str, help="Path to save the merged model")
    parser.add_argument("--merge_all", action="store_true", help="Merge all discovered model pairs")
    parser.add_argument("--model_name", type=str, help="Name of the model for single merge")
    
    args = parser.parse_args()
    
    if args.merge_all:
        # Merge all available models
        logger.info("🚀 Starting batch merge of all student models")
        success = merge_all_models()
        exit(0 if success else 1)
    
    elif args.base_model_path and args.peft_model_path and args.output_model_path:
        # Single model merge
        model_name = args.model_name or "custom-model"
        success = merge_peft_model(
            args.base_model_path, 
            args.peft_model_path, 
            args.output_model_path,
            model_name
        )
        exit(0 if success else 1)
    
    else:
        # Interactive mode - show available options
        logger.info("🔍 Discovering available models...")
        model_pairs = discover_model_pairs()
        
        if not model_pairs:
            logger.error("❌ No model pairs found!")
            logger.info("💡 Use --merge_all to merge all available models")
            logger.info("💡 Or specify --base_model_path, --peft_model_path, --output_model_path for single merge")
            exit(1)
        
        logger.info("📋 Available models for merging:")
        for pair in model_pairs:
            logger.info(f"  - {pair['name']}")
        
        logger.info("💡 Run with --merge_all to merge all models")
        logger.info("💡 Or specify individual paths for single model merge")

if __name__ == "__main__":
    main()