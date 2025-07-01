import os
import json
import torch
import time
import psutil
import logging
import threading
import subprocess
import csv
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import gc

# Setup logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CSV logging setup
csv_log_file = 'resource_usage_log.csv'
csv_headers = [
    'timestamp', 'model_name', 'stage', 'batch_number', 'processed_instructions',
    'cpu_percent', 'ram_percent', 'ram_used_gb', 'ram_total_gb',
    'gpu_util_percent', 'gpu_memory_used_gb', 'gpu_memory_total_gb', 'gpu_memory_percent',
    'throughput_instr_per_sec', 'elapsed_time_sec'
]

def init_csv_logger():
    """Initialize CSV logger with headers"""
    with open(csv_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

def log_to_csv(model_name, stage, batch_number=None, processed_instructions=None, throughput=None, elapsed_time=None):
    """Log system resources to CSV"""
    timestamp = datetime.now().isoformat()
    
    # Get system resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    ram_percent = memory.percent
    ram_used_gb = memory.used / 1024**3
    ram_total_gb = memory.total / 1024**3
    
    # Get GPU resources
    gpu_info = get_gpu_utilization()
    if gpu_info:
        gpu_util_percent = gpu_info['gpu_util']
        gpu_memory_used_gb = gpu_info['memory_used']
        gpu_memory_total_gb = gpu_info['memory_total']
        gpu_memory_percent = gpu_info['memory_util']
    else:
        gpu_util_percent = gpu_memory_used_gb = gpu_memory_total_gb = gpu_memory_percent = 'N/A'
    
    # Write to CSV
    with open(csv_log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, model_name, stage, batch_number, processed_instructions,
            cpu_percent, ram_percent, ram_used_gb, ram_total_gb,
            gpu_util_percent, gpu_memory_used_gb, gpu_memory_total_gb, gpu_memory_percent,
            throughput, elapsed_time
        ])

def get_gpu_utilization():
    """Get GPU utilization using nvidia-ml-py or nvidia-smi fallback"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'gpu_util': util.gpu,
            'memory_util': util.memory,
            'memory_used': memory_info.used / 1024**3,
            'memory_total': memory_info.total / 1024**3
        }
    except:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                line = result.stdout.strip().split('\n')[0]
                gpu_util, mem_used, mem_total = line.split(', ')
                return {
                    'gpu_util': int(gpu_util),
                    'memory_util': int(float(mem_used) / float(mem_total) * 100),
                    'memory_used': float(mem_used) / 1024,
                    'memory_total': float(mem_total) / 1024
                }
        except:
            pass
    return None

def start_gpu_monitor(interval=5):
    """Start background GPU monitoring thread"""
    def monitor():
        while getattr(threading.current_thread(), "do_run", True):
            gpu_info = get_gpu_utilization()
            if gpu_info:
                logger.info(f"🖥️  GPU: {gpu_info['gpu_util']}% util | "
                           f"VRAM: {gpu_info['memory_used']:.1f}/{gpu_info['memory_total']:.1f}GB "
                           f"({gpu_info['memory_util']}%)")
            time.sleep(interval)
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.do_run = True
    monitor_thread.start()
    return monitor_thread

# Updated list of distilled models with fallback to HuggingFace models
distilled_models = {
    "qwen2.5-1.5b": {
        "base_model": "StudentModels/qwen_models/2.5-1.5B/",
        "adapter": "StudentModels/Adapters/qwen2.5-1.5b-lora/",
        "hf_fallback": "Qwen/Qwen2.5-1.5B"  # Hugging Face fallback
    },
    "qwen2.5-7b": {
        "base_model": "StudentModels/qwen_models/2.5-7B/",
        "adapter": "StudentModels/Adapters/qwen2.5-7b-qlora/checkpoint-4122/",
        "hf_fallback": "Qwen/Qwen2.5-7B"  # Hugging Face fallback
    }
}

# Path to the dataset - Updated to use existing files
jsonl_file_path = "Dataset/codealpaca_gsm8k_30k.jsonl"  # Use the main dataset file

# Function to load instructions from the JSONL file
def load_instructions(jsonl_file_path):
    instructions = []
    
    # Check if file exists and provide alternatives
    if not os.path.exists(jsonl_file_path):
        logger.error(f"Dataset file not found: {jsonl_file_path}")
        
        # Try alternative paths with files that are more likely to exist
        alternative_paths = [
            "Dataset/codealpaca_gsm8k_30k.jsonl",  # Main combined dataset
            "Dataset/categorized_split_data_stratified_train.jsonl",  # Training split
            "Dataset/categorized_split_data_stratified_test.jsonl",  # Test split
            "Dataset/devstral_inference.jsonl",  # Teacher inference results
            "Scripts/Inference/categorized_split_data_stratified_test.jsonl",  # Local copy
            "./categorized_split_data_stratified_test.jsonl",  # Current directory
            "../Dataset/codealpaca_gsm8k_30k.jsonl"  # One level up
        ]
        
        logger.info("Searching for alternative dataset files...")
        for alt_path in alternative_paths:
            logger.info(f"Checking: {alt_path}")
            if os.path.exists(alt_path):
                logger.info(f"✅ Found alternative dataset: {alt_path}")
                jsonl_file_path = alt_path
                break
        else:
            logger.error("❌ No dataset file found. Please check the following paths:")
            for path in alternative_paths:
                logger.error(f"  - {path} {'✅' if os.path.exists(path) else '❌'}")
            
            # List all .jsonl files in current directory and parent directories
            logger.info("Available .jsonl files in current directory:")
            for file in os.listdir("."):
                if file.endswith(".jsonl"):
                    logger.info(f"  - {file}")
            
            if os.path.exists("Dataset"):
                logger.info("Available .jsonl files in Dataset:")
                for file in os.listdir("Dataset"):
                    if file.endswith(".jsonl"):
                        logger.info(f"  - Dataset/{file}")
            
            raise FileNotFoundError("No dataset file available for inference")
    
    logger.info(f"📂 Loading instructions from: {jsonl_file_path}")
    with open(jsonl_file_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            try:
                data = json.loads(line.strip())
                if "instruction" in data:
                    instructions.append(data["instruction"])
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  Skipping malformed JSON on line {line_num}: {e}")
                continue
    
    logger.info(f"✅ Successfully loaded {len(instructions)} instructions")
    return instructions

def infer_qwen_0_5b_optimized(model_paths, instructions):
    """Optimized inference for Qwen 0.5B - smallest model, very aggressive batching"""
    logger.info("🚀 Starting optimized inference for Qwen 0.5B")
    model_name = "qwen2.5-0.5b"
    
    # Very aggressive settings for small model - 0.5B is tiny compared to 24B
    config = {
        "batch_size": 64,  # Increased for 100% GPU utilization
        "max_new_tokens": 64,
        "temperature": 0.6,
        "top_p": 0.9,
        "use_flash_attention": False,
        "use_torch_compile": False  # Disabled for stability
    }
    
    return perform_model_specific_inference(model_name, model_paths, instructions, config)

def infer_qwen_1_5b_optimized(model_paths, instructions):
    """Optimized inference for Qwen 1.5B - medium model, aggressive batching"""
    logger.info("🚀 Starting optimized inference for Qwen 1.5B")
    model_name = "qwen2.5-1.5b"  # Fixed model name
    
    # Aggressive settings for medium model
    config = {
        "batch_size": 32,  # Increased for 100% GPU utilization
        "max_new_tokens": 96,
        "temperature": 0.7,
        "top_p": 0.95,
        "use_flash_attention": False,
        "use_torch_compile": False  # Disabled for stability
    }
    
    return perform_model_specific_inference(model_name, model_paths, instructions, config)

def infer_qwen_7b_optimized(model_paths, instructions):
    """Optimized inference for Qwen 7B - large model, memory-conscious"""
    logger.info("🚀 Starting optimized inference for Qwen 7B")
    model_name = "qwen2.5-7b"
    
    # Conservative settings for large model
    config = {
        "batch_size": 16,  # Increased for better GPU utilization
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_flash_attention": False,
        "use_torch_compile": False,  # Disabled for stability
        "gradient_checkpointing": True
    }
    
    return perform_model_specific_inference(model_name, model_paths, instructions, config)

def perform_model_specific_inference(model_name, model_paths, instructions, config):
    """Enhanced inference function with model-specific optimizations"""
    output_file = f"{model_name}_output.jsonl"
    start_time = time.time()
    
    # Log initial system state
    logger.info(f"🔄 Starting inference for {model_name}")
    log_system_resources()
    log_gpu_memory()
    log_to_csv(model_name, "initialization")
    
    # Start GPU monitoring
    monitor_thread = start_gpu_monitor(interval=3)
    
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
        
        # Determine which model to load (local vs HuggingFace)
        base_model_path = model_paths["base_model"]
        if not os.path.exists(base_model_path) and "hf_fallback" in model_paths:
            logger.warning(f"⚠️  Local model not found at {base_model_path}")
            logger.info(f"🔄 Using HuggingFace fallback: {model_paths['hf_fallback']}")
            base_model_path = model_paths["hf_fallback"]
            # Skip adapter loading for HuggingFace models since we don't have local adapters
            use_adapter = False
        else:
            use_adapter = os.path.exists(model_paths.get("adapter", ""))
            if not use_adapter:
                logger.warning(f"⚠️  Adapter not found at {model_paths.get('adapter', 'N/A')}")
        
        # Load tokenizer with optimizations
        logger.info(f"📥 Loading tokenizer from {base_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left"  # Better for batch processing
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with enhanced settings
        logger.info(f"📥 Loading model from {base_model_path}...")
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
        
        if config.get("use_flash_attention", False):
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        
        # Load LoRA adapter if available
        if use_adapter:
            logger.info(f"📥 Loading LoRA adapter from {model_paths['adapter']}...")
            model = PeftModel.from_pretrained(model, model_paths["adapter"])
        else:
            logger.info("⚠️  Skipping adapter loading - using base model only")
        
        model.eval()
        
        # Apply gradient checkpointing if specified
        if config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        
        # Compile model for speed (PyTorch 2.0+)
        if config.get("use_torch_compile", False) and hasattr(torch, 'compile'):
            logger.info("⚡ Compiling model for optimized inference...")
            model = torch.compile(model, mode="max-autotune")
        
        model_load_time = time.time() - model_start
        logger.info(f"✅ Model loaded in {model_load_time:.2f}s")
        
        # Log system state after model loading
        log_system_resources()
        log_gpu_memory()
        log_to_csv(model_name, "model_loaded", elapsed_time=model_load_time)
        
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
                    log_system_resources()
                    log_gpu_memory()
                    log_to_csv(model_name, "batch_processing", batch_number=batch_number, 
                              processed_instructions=processed, throughput=throughput)
                
                # Log brief progress every 50 batches
                elif (batch_start // batch_size) % 50 == 0:
                    throughput = processed / (time.time() - start_time)
                    batch_number = batch_start // batch_size
                    logger.info(f"📊 Progress: {processed}/{total_instructions} | "
                               f"Throughput: {throughput:.2f} instr/sec")
                    log_to_csv(model_name, "batch_progress", batch_number=batch_number, 
                              processed_instructions=processed, throughput=throughput)
        
    finally:
        # Stop monitoring and cleanup
        monitor_thread.do_run = False
        
        # Log final system state before cleanup
        logger.info(f"🧹 Cleaning up {model_name}...")
        log_system_resources()
        log_gpu_memory()
        log_to_csv(model_name, "cleanup_start")
        
        # Cleanup
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log system state after cleanup
        logger.info(f"✅ Cleanup completed for {model_name}")
        log_system_resources()
        log_gpu_memory()
        log_to_csv(model_name, "cleanup_complete")
    
    total_time = time.time() - start_time
    throughput = len(instructions) / total_time
    
    logger.info(f"✅ {model_name} completed in {total_time:.2f}s | "
               f"Throughput: {throughput:.2f} instr/sec")
    
    # Log final completion stats
    log_to_csv(model_name, "completion", processed_instructions=len(instructions), 
              throughput=throughput, elapsed_time=total_time)
    
    return output_file

def process_enhanced_batch(model, tokenizer, instructions, config):
    """Enhanced batch processing with optimized GPU utilization"""
    batch_results = []
    
    # Process all instructions in the batch with SemiQwenn system prompt
    system_prompt = """You are SemiQwenn, a helpful agentic model trained by Semiqolonn and using distillation SFT on DevStral-24B. You can interact with a computer to solve tasks."""
    
    prompts = [f"{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Response:\n" 
              for instruction in instructions]
    
    try:
        # Tokenize batch with padding for efficient processing
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,  # Pad to same length for batch processing
            add_special_tokens=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with optimized settings using PyTorch autocast
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
                    num_beams=1,
                    output_scores=False,
                    return_dict_in_generate=False
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
        logger.error(f"❌ Batch processing error: {str(e)[:100]}")
        # Fallback to individual processing
        for instruction in instructions:
            batch_results.append(f"ERROR: {str(e)[:50]}")
    
    return batch_results

# Updated function to handle models using transformers and PEFT with optimizations
def perform_inference_optimized(model_name, model_paths, instructions):
    output_file = f"{model_name}_output.jsonl"
    logger.info(f"Starting inference for model: {model_name}")
    
    # Get model-specific configuration
    config = get_model_specific_config(model_name)
    batch_size = config["batch_size"]
    
    # Check GPU availability and log system state
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Set memory fraction for better utilization
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    log_system_resources()
    log_gpu_memory()

    # Setup optimized quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load tokenizer with optimizations
    logger.info(f"Loading tokenizer from {model_paths['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_paths["base_model"],
        trust_remote_code=True,
        use_fast=True  # Use fast tokenizer for speed
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with optimizations
    logger.info(f"Loading base model from {model_paths['base_model']}")
    start_time = time.time()
    
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "use_cache": True
    }
    
    # Add flash attention if supported
    if config.get("use_flash_attention", False):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_paths["base_model"],
        **model_kwargs
    )
    
    # Enable gradient checkpointing for large models
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    log_gpu_memory()

    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from {model_paths['adapter']}")
    start_time = time.time()
    model = PeftModel.from_pretrained(model, model_paths["adapter"])
    adapter_time = time.time() - start_time
    logger.info(f"LoRA adapter loaded in {adapter_time:.2f} seconds")
    
    # Set model to evaluation mode and optimize
    model.eval()
    
    # Compile model for faster inference (PyTorch 2.0+)
    # Disabled due to compatibility issues with quantized models and PEFT
    # if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
    #     logger.info("Compiling model for optimized inference...")
    #     model = torch.compile(model, mode="reduce-overhead")

    log_gpu_memory()

    # Process in batches for better GPU utilization
    total_instructions = len(instructions)
    processed = 0
    
    with open(output_file, "w") as output:
        # Process instructions in batches
        for batch_start in tqdm(range(0, total_instructions, batch_size), 
                               desc=f"Processing {model_name} (batch_size={batch_size})"):
            batch_end = min(batch_start + batch_size, total_instructions)
            batch_instructions = instructions[batch_start:batch_end]
            
            # Process batch
            batch_results = process_batch(model, tokenizer, batch_instructions, config)
            
            # Write results
            for instruction, output_text in zip(batch_instructions, batch_results):
                output_data = {
                    "instruction": instruction,
                    "output": output_text
                }
                output.write(json.dumps(output_data) + "\n")
            
            processed += len(batch_instructions)
            
            # Log progress every 100 batches
            if (batch_start // batch_size) % 100 == 0:
                log_gpu_memory()
                logger.info(f"Processed {processed}/{total_instructions} instructions")

    logger.info(f"Inference completed for model: {model_name}. Results saved to {output_file}")
    
    # Clean up GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    log_gpu_memory()

def process_batch(model, tokenizer, instructions, config):
    """Process a batch of instructions efficiently"""
    batch_results = []
    
    # SemiQwenn system prompt
    system_prompt = """You are SemiQwenn, a helpful agentic model trained by Semiqolonn and using distillation SFT on DevStral-24B. You can interact with a computer to solve tasks."""
    
    for instruction in instructions:
        prompt = f"{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        
        try:
            # Tokenize input with optimized settings
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,  # Reduced for speed
                padding=False
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response with model-specific optimizations using PyTorch autocast
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=config["max_new_tokens"],
                        temperature=config["temperature"],
                        do_sample=True,
                        top_p=config["top_p"],
                        top_k=config["top_k"],
                        repetition_penalty=1.05,  # Reduced for speed
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1  # Greedy for speed
                    )
            
            # Decode response
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            output_text = response.strip()
            
            # Clean up response
            for stop_seq in ["###", "</s>", "### Instruction:", "### Response:"]:
                if stop_seq in output_text:
                    output_text = output_text.split(stop_seq)[0].strip()
                    
        except Exception as e:
            output_text = f"ERROR: {str(e)[:100]}"
            logger.error(f"Error processing instruction: {str(e)[:100]}")

        batch_results.append(output_text)
    
    return batch_results

def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_utilization = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 'N/A'
            
            logger.info(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, "
                       f"{memory_total:.2f}GB total, Utilization: {gpu_utilization}%")

def log_system_resources():
    """Log system CPU and memory usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU: {cpu_percent}%, RAM: {memory.percent}% ({memory.used/1024**3:.2f}GB/{memory.total/1024**3:.2f}GB)")

def get_optimal_batch_size(model_name):
    """Get optimal batch size based on model size"""
    batch_sizes = {
        "qwen2.5-0.5b": 8,  # Smaller model, can handle larger batches
        "qwen1.5b": 4,      # Medium model
        "qwen2.5-7b": 1     # Large model, single sample at a time
    }
    return batch_sizes.get(model_name, 1)

def get_model_specific_config(model_name):
    """Get model-specific optimization configs"""
    configs = {
        "qwen2.5-0.5b": {
            "max_new_tokens": 96,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "batch_size": 64,  # Increased for 100% GPU utilization
            "use_flash_attention": False,
            "gradient_checkpointing": False
        },
        "qwen1.5b": {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "batch_size": 32,  # Increased for 100% GPU utilization
            "use_flash_attention": False,
            "gradient_checkpointing": False
        },
        "qwen2.5-7b": {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "batch_size": 16,  # Increased for better GPU utilization
            "use_flash_attention": False,
            "gradient_checkpointing": True
        }
    }
    return configs.get(model_name, configs["qwen1.5b"])

# Main script
if __name__ == "__main__":
    # Initialize CSV logger
    init_csv_logger()
    
    # Load instructions from the JSONL file
    instructions = load_instructions(jsonl_file_path)
    logger.info(f"Loaded {len(instructions)} instructions from {jsonl_file_path}")

    # Log initial system state
    log_system_resources()
    log_gpu_memory()
    log_to_csv("global", "startup")

    # Define model-specific inference functions for optimal performance
    model_inference_functions = {
        # "qwen2.5-0.5b": infer_qwen_0_5b_optimized,  # Commented out - already inferred
        "qwen2.5-1.5b": infer_qwen_1_5b_optimized,
        "qwen2.5-7b": infer_qwen_7b_optimized
    }

    # Perform inference for each model with model-specific optimizations
    for model_name, model_paths in distilled_models.items():
        logger.info(f"=" * 60)
        logger.info(f"🚀 Starting optimized inference for {model_name}")
        logger.info(f"=" * 60)
        
        start_time = time.time()
        
        # Use model-specific inference function if available
        if model_name in model_inference_functions:
            output_file = model_inference_functions[model_name](model_paths, instructions)
        else:
            # Fallback to generic function
            perform_inference_optimized(model_name, model_paths, instructions)
        
        total_time = time.time() - start_time
        
        logger.info(f"🎯 Total time for {model_name}: {total_time:.2f} seconds")
        logger.info(f"⚡ Average time per instruction: {total_time/len(instructions):.3f} seconds")
        logger.info(f"🚄 Throughput: {len(instructions)/total_time:.2f} instructions/second")
        logger.info("")
        
        # Log model completion to CSV
        log_to_csv(model_name, "model_complete", processed_instructions=len(instructions),
                  throughput=len(instructions)/total_time, elapsed_time=total_time)

    logger.info("🎉 All model inference completed!")
    log_to_csv("global", "all_complete")
