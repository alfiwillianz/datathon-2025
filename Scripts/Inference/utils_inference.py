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
import gc

# Setup logging with more detailed formatting
def setup_logger(model_name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{model_name}_inference_log.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# CSV logging setup
def init_csv_logger(model_name):
    """Initialize CSV logger with headers"""
    csv_log_file = f'{model_name}_resource_usage_log.csv'
    csv_headers = [
        'timestamp', 'model_name', 'stage', 'batch_number', 'processed_instructions',
        'cpu_percent', 'ram_percent', 'ram_used_gb', 'ram_total_gb',
        'gpu_util_percent', 'gpu_memory_used_gb', 'gpu_memory_total_gb', 'gpu_memory_percent',
        'throughput_instr_per_sec', 'elapsed_time_sec'
    ]
    
    with open(csv_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    return csv_log_file

def log_to_csv(csv_log_file, model_name, stage, batch_number=None, processed_instructions=None, throughput=None, elapsed_time=None):
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

def start_gpu_monitor(logger, interval=5):
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

def load_instructions(jsonl_file_path):
    """Load instructions from the JSONL file"""
    instructions = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if "instruction" in data:
                instructions.append(data["instruction"])
    return instructions

def log_gpu_memory(logger):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            logger.info(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, "
                       f"{memory_total:.2f}GB total")

def log_system_resources(logger):
    """Log system CPU and memory usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU: {cpu_percent}%, RAM: {memory.percent}% ({memory.used/1024**3:.2f}GB/{memory.total/1024**3:.2f}GB)")

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
        # Fallback to individual processing
        for instruction in instructions:
            batch_results.append(f"ERROR: {str(e)[:50]}")
    
    return batch_results

def process_base_model_batch(model, tokenizer, instructions, config):
    """Process batch for base models without special system prompts"""
    batch_results = []
    
    # Simple instruction format for base models
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
        # Fallback to individual processing
        for instruction in instructions:
            batch_results.append(f"ERROR: {str(e)[:50]}")
    
    return batch_results
