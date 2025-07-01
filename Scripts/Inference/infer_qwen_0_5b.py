import os
import json
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import gc
from utils_inference import (
    setup_logger, init_csv_logger, log_to_csv, start_gpu_monitor,
    load_instructions, log_gpu_memory, log_system_resources, process_enhanced_batch
)

def infer_qwen_0_5b_optimized():
    """Optimized inference for Qwen 0.5B - smallest model, very aggressive batching"""
    model_name = "qwen2.5-0.5b"
    logger = setup_logger(model_name)
    csv_log_file = init_csv_logger(model_name)
    
    logger.info("🚀 Starting optimized inference for Qwen 0.5B")
    
    # Model configuration
    model_paths = {
        "base_model": "StudentModels/qwen_models/2.5-0.5B/",
        "adapter": "StudentModels/Adapters/qwen2.5-0.5b-lora/checkpoint-4122/"
    }
    
    # Very aggressive settings for small model - 0.5B is tiny
    config = {
        "batch_size": 64,  # Increased for 100% GPU utilization
        "max_new_tokens": 64,
        "temperature": 0.6,
        "top_p": 0.9,
        "use_flash_attention": False,
        "use_torch_compile": False
    }
    
    # Load instructions
    jsonl_file_path = "Dataset/categorized_split_data_stratified_test.jsonl"
    instructions = load_instructions(jsonl_file_path)
    logger.info(f"Loaded {len(instructions)} instructions")
    
    output_file = f"{model_name}_output.jsonl"
    start_time = time.time()
    
    # Log initial system state
    logger.info(f"🔄 Starting inference for {model_name}")
    log_system_resources(logger)
    log_gpu_memory(logger)
    log_to_csv(csv_log_file, model_name, "initialization")
    
    # Start GPU monitoring
    monitor_thread = start_gpu_monitor(logger, interval=3)
    
    try:
        # GPU setup and optimization
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔧 Device: {device}")
        
        if torch.cuda.is_available():
            # Clear cache and optimize GPU settings
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Set optimal memory allocation strategy - more aggressive
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Enhanced quantization for better performance
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8
        )
        
        # Load tokenizer with optimizations
        logger.info(f"📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_paths["base_model"],
            trust_remote_code=True,
            use_fast=True,
            padding_side="left"  # Better for batch processing
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with enhanced settings
        logger.info(f"📥 Loading model...")
        model_start = time.time()
        
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_cache": True,
            "max_memory": {0: "15GB"}  # More aggressive memory usage for 16GB GPU
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_paths["base_model"],
            **model_kwargs
        )
        
        # Load LoRA adapter
        logger.info(f"📥 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_paths["adapter"])
        model.eval()
        
        model_load_time = time.time() - model_start
        logger.info(f"✅ Model loaded in {model_load_time:.2f}s")
        
        # Log system state after model loading
        log_system_resources(logger)
        log_gpu_memory(logger)
        log_to_csv(csv_log_file, model_name, "model_loaded", elapsed_time=model_load_time)
        
        # Process instructions with enhanced batching
        total_instructions = len(instructions)
        batch_size = config["batch_size"]
        processed = 0
        
        logger.info(f"🔄 Processing {total_instructions} instructions with batch_size={batch_size}")
        
        with open(output_file, "w") as output:
            for batch_start in tqdm(range(0, total_instructions, batch_size), 
                                   desc=f"Inferencing {model_name}"):
                batch_end = min(batch_start + batch_size, total_instructions)
                batch_instructions = instructions[batch_start:batch_end]
                
                # Process batch with enhanced settings
                batch_results = process_enhanced_batch(
                    model, tokenizer, batch_instructions, config
                )
                
                # Write results
                for instruction, output_text in zip(batch_instructions, batch_results):
                    output_data = {
                        "instruction": instruction,
                        "output": output_text
                    }
                    output.write(json.dumps(output_data) + "\n")
                
                processed += len(batch_instructions)
                
                # Log progress and resources every 10 batches
                if (batch_start // batch_size) % 10 == 0:
                    throughput = processed / (time.time() - start_time)
                    batch_number = batch_start // batch_size
                    logger.info(f"📊 Progress: {processed}/{total_instructions} | "
                               f"Throughput: {throughput:.2f} instr/sec")
                    log_system_resources(logger)
                    log_gpu_memory(logger)
                    log_to_csv(csv_log_file, model_name, "batch_processing", batch_number=batch_number, 
                              processed_instructions=processed, throughput=throughput)
        
    finally:
        # Stop monitoring and cleanup
        monitor_thread.do_run = False
        
        # Log final system state before cleanup
        logger.info(f"🧹 Cleaning up {model_name}...")
        log_system_resources(logger)
        log_gpu_memory(logger)
        log_to_csv(csv_log_file, model_name, "cleanup_start")
        
        # Cleanup
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log system state after cleanup
        logger.info(f"✅ Cleanup completed for {model_name}")
        log_system_resources(logger)
        log_gpu_memory(logger)
        log_to_csv(csv_log_file, model_name, "cleanup_complete")
    
    total_time = time.time() - start_time
    throughput = len(instructions) / total_time
    
    logger.info(f"✅ {model_name} completed in {total_time:.2f}s | "
               f"Throughput: {throughput:.2f} instr/sec")
    
    # Log final completion stats
    log_to_csv(csv_log_file, model_name, "completion", processed_instructions=len(instructions), 
              throughput=throughput, elapsed_time=total_time)
    
    return output_file

if __name__ == "__main__":
    infer_qwen_0_5b_optimized()
