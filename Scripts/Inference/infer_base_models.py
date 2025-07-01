import os
import json
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import gc
from utils_inference import (
    setup_logger, init_csv_logger, log_to_csv, start_gpu_monitor,
    load_instructions, log_gpu_memory, log_system_resources, process_enhanced_batch
)

def discover_base_models():
    """Discover all available base student models"""
    base_models = {}
    base_path = "StudentModels/qwen_models/"
    
    if not os.path.exists(base_path):
        logger.error(f"StudentModels directory not found: {base_path}")
        return {}
    
    # Look for model directories
    for item in os.listdir(base_path):
        model_path = os.path.join(base_path, item)
        if os.path.isdir(model_path):
            # Check if it's a valid model directory (contains config.json)
            if os.path.exists(os.path.join(model_path, "config.json")):
                model_name = f"qwen-base-{item.lower()}"
                base_models[model_name] = {
                    "base_model": model_path,
                    "size": item
                }
    
    return base_models

def get_base_model_config(model_size):
    """Get optimized config for base models based on size"""
    configs = {
        "2.5-0.5B": {
            "batch_size": 64,
            "max_new_tokens": 96,
            "temperature": 0.7,
            "top_p": 0.9,
            "memory_fraction": 0.95
        },
        "2.5-1.5B": {
            "batch_size": 32,
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "memory_fraction": 0.90
        },
        "2.5-7B": {
            "batch_size": 16,
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "memory_fraction": 0.85,
            "gradient_checkpointing": True
        }
    }
    
    # Default config for unknown sizes
    default_config = {
        "batch_size": 16,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "memory_fraction": 0.85
    }
    
    return configs.get(model_size, default_config)

def process_base_model_batch(model, tokenizer, instructions, config):
    """Process batch for base models (no special system prompt)"""
    batch_results = []
    
    # Simple prompt format for base models
    prompts = [f"### Instruction:\n{instruction}\n\n### Response:\n" 
              for instruction in instructions]
    
    try:
        # Tokenize batch with padding
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            add_special_tokens=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with optimized settings
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    do_sample=True,
                    top_p=config["top_p"],
                    top_k=40,
                    repetition_penalty=1.03,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1
                )
        
        # Decode responses
        for i, output in enumerate(outputs):
            response = tokenizer.decode(
                output[inputs['input_ids'][i].shape[0]:], 
                skip_special_tokens=True
            )
            
            # Clean response
            output_text = response.strip()
            for stop_seq in ["###", "</s>", "### Instruction:", "### Response:"]:
                if stop_seq in output_text:
                    output_text = output_text.split(stop_seq)[0].strip()
            
            batch_results.append(output_text)
    
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        for instruction in instructions:
            batch_results.append(f"ERROR: {str(e)[:50]}")
    
    return batch_results

def infer_base_model(model_name, model_path, instructions):
    """Run inference on a base model"""
    logger.info(f"🚀 Starting base model inference: {model_name}")
    
    # Get model size from path
    model_size = model_path.split('/')[-2] if '/' in model_path else model_path
    config = get_base_model_config(model_size)
    
    output_file = f"{model_name}_base_output.jsonl"
    start_time = time.time()
    
    # Log initial state
    log_system_resources(logger)
    log_gpu_memory(logger)
    log_to_csv(csv_log_file, model_name, "initialization")
    
    # Start GPU monitoring
    monitor_thread = start_gpu_monitor(logger, interval=3)
    
    try:
        # GPU setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔧 Device: {device}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])
            
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8
        )
        
        # Load tokenizer
        logger.info(f"📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        logger.info(f"📥 Loading base model...")
        model_start = time.time()
        
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_cache": True
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        model.eval()
        
        # Apply gradient checkpointing for large models
        if config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        
        model_load_time = time.time() - model_start
        logger.info(f"✅ Base model loaded in {model_load_time:.2f}s")
        
        log_system_resources(logger)
        log_gpu_memory(logger)
        log_to_csv(csv_log_file, model_name, "model_loaded", elapsed_time=model_load_time)
        
        # Process instructions
        total_instructions = len(instructions)
        batch_size = config["batch_size"]
        processed = 0
        
        logger.info(f"🔄 Processing {total_instructions} instructions with batch_size={batch_size}")
        
        with open(output_file, "w") as output:
            for batch_start in tqdm(range(0, total_instructions, batch_size), 
                                   desc=f"Base inference {model_name}"):
                batch_end = min(batch_start + batch_size, total_instructions)
                batch_instructions = instructions[batch_start:batch_end]
                
                # Process batch
                batch_results = process_base_model_batch(
                    model, tokenizer, batch_instructions, config
                )
                
                # Write results
                for instruction, output_text in zip(batch_instructions, batch_results):
                    output_data = {
                        "instruction": instruction,
                        "output": output_text,
                        "model_type": "base",
                        "model_name": model_name
                    }
                    output.write(json.dumps(output_data) + "\n")
                
                processed += len(batch_instructions)
                
                # Log progress every 10 batches
                if (batch_start // batch_size) % 10 == 0:
                    throughput = processed / (time.time() - start_time)
                    batch_number = batch_start // batch_size
                    logger.info(f"📊 Progress: {processed}/{total_instructions} | "
                               f"Throughput: {throughput:.2f} instr/sec")
                    log_system_resources(logger)
                    log_gpu_memory(logger)
                    log_to_csv(csv_log_file, model_name, "batch_processing", 
                              batch_number=batch_number, processed_instructions=processed, 
                              throughput=throughput)
        
    finally:
        # Cleanup
        monitor_thread.do_run = False
        
        logger.info(f"🧹 Cleaning up {model_name}...")
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"✅ Cleanup completed for {model_name}")
        log_to_csv(csv_log_file, model_name, "cleanup_complete")
    
    total_time = time.time() - start_time
    throughput = len(instructions) / total_time
    
    logger.info(f"✅ {model_name} completed in {total_time:.2f}s | "
               f"Throughput: {throughput:.2f} instr/sec")
    
    log_to_csv(csv_log_file, model_name, "completion", 
              processed_instructions=len(instructions), 
              throughput=throughput, elapsed_time=total_time)
    
    return output_file

def main():
    """Main function to run inference on all base models"""
    global logger, csv_log_file
    
    # Setup logging
    logger = setup_logger("base_models")
    csv_log_file = init_csv_logger("base_models")
    
    logger.info("🚀 Starting base model inference for all student models")
    
    # Discover available models
    base_models = discover_base_models()
    
    if not base_models:
        logger.error("❌ No base models found!")
        logger.info("Please ensure StudentModels/qwen_models/ contains model directories")
        return
    
    logger.info(f"📋 Found {len(base_models)} base models:")
    for model_name, model_info in base_models.items():
        logger.info(f"  - {model_name}: {model_info['base_model']}")
    
    # Load instructions
    jsonl_file_path = "Dataset/Splitted/categorized_split_data_stratified_test.jsonl"
    instructions = load_instructions(jsonl_file_path)
    logger.info(f"📂 Loaded {len(instructions)} instructions")
    
    # Log initial system state
    log_system_resources(logger)
    log_gpu_memory(logger)
    log_to_csv(csv_log_file, "global", "startup")
    
    # Run inference for each model
    results = {}
    total_start_time = time.time()
    
    # Sort models by size (smallest first for better resource management)
    sorted_models = sorted(base_models.items(), 
                          key=lambda x: x[1]['size'])
    
    for model_name, model_info in sorted_models:
        logger.info("=" * 60)
        logger.info(f"🎯 Starting inference: {model_name}")
        logger.info(f"📁 Model path: {model_info['base_model']}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            output_file = infer_base_model(
                model_name, 
                model_info['base_model'], 
                instructions
            )
            
            elapsed_time = time.time() - start_time
            throughput = len(instructions) / elapsed_time
            
            results[model_name] = {
                "success": True,
                "output_file": output_file,
                "elapsed_time": elapsed_time,
                "throughput": throughput
            }
            
            logger.info(f"✅ {model_name} completed successfully")
            logger.info(f"⏱️  Time: {elapsed_time:.2f}s")
            logger.info(f"⚡ Throughput: {throughput:.2f} instr/sec")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ {model_name} failed: {str(e)}")
            
            results[model_name] = {
                "success": False,
                "error": str(e),
                "elapsed_time": elapsed_time
            }
        
        # Brief pause between models
        time.sleep(3)
    
    # Summary
    total_time = time.time() - total_start_time
    successful = sum(1 for r in results.values() if r["success"])
    total_models = len(results)
    
    logger.info("=" * 60)
    logger.info("📊 BASE MODEL INFERENCE SUMMARY")
    logger.info("=" * 60)
    
    for model_name, result in results.items():
        if result["success"]:
            logger.info(f"✅ {model_name}: {result['elapsed_time']:.2f}s "
                       f"({result['throughput']:.2f} instr/sec)")
        else:
            logger.info(f"❌ {model_name}: FAILED - {result.get('error', 'Unknown error')}")
    
    logger.info(f"🎯 Results: {successful}/{total_models} models completed successfully")
    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    
    if successful > 0:
        avg_throughput = sum(r["throughput"] for r in results.values() 
                           if r["success"]) / successful
        logger.info(f"⚡ Average throughput: {avg_throughput:.2f} instr/sec")
    
    if successful == total_models:
        logger.info("🎉 All base models completed successfully!")
    else:
        logger.warning(f"⚠️  {total_models - successful} models failed")
    
    log_to_csv(csv_log_file, "global", "all_complete", 
              processed_instructions=len(instructions) * successful,
              elapsed_time=total_time)

if __name__ == "__main__":
    main()
