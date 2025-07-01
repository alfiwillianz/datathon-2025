import os
import sys
import logging
import json

# Set environment variables to avoid TensorFlow warnings and compatibility issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disable TensorFlow entirely for transformers to avoid version conflicts
os.environ['USE_TF'] = 'NO'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Setup logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Check for TensorFlow Nightly and warn user
def check_tensorflow_version():
    """Check if TensorFlow Nightly is installed and suggest fix"""
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        logging.info(f"🔍 TensorFlow version detected: {tf_version}")
        
        if 'nightly' in tf_version.lower() or 'dev' in tf_version.lower():
            logging.warning("⚠️  TensorFlow Nightly detected!")
            logging.warning("This may cause compatibility issues with transformers library")
            logging.info("💡 Recommended fix:")
            logging.info("   pip uninstall tensorflow-nightly")
            logging.info("   pip install tensorflow==2.13.0")
            logging.info("   OR set USE_TF=NO to disable TensorFlow in transformers")
            
        return True
    except ImportError:
        logging.info("✅ TensorFlow not found - this is fine for PyTorch-only training")
        return True
    except Exception as e:
        logging.warning(f"Could not check TensorFlow version: {e}")
        return True

# Check TensorFlow before importing transformers
check_tensorflow_version()

try:
    # Force transformers to use PyTorch only
    import transformers
    transformers.utils.is_tf_available = lambda: False
    transformers.utils.is_tensorflow_probability_available = lambda: False
    
    # Import transformers components individually to catch specific import errors
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
    from transformers import TrainingArguments
    from transformers import DataCollatorForLanguageModeling
    from transformers import TrainerCallback
    
    # Try to import Trainer separately as it's causing the issue
    try:
        from transformers import Trainer
        logging.info("✅ All transformers imports successful")
    except ImportError as e:
        logging.error(f"❌ Failed to import Trainer: {e}")
        logging.info("🔧 Attempting workaround...")
        # Try alternative import approach
        import transformers
        Trainer = transformers.Trainer
        logging.info("✅ Trainer imported via workaround")
        
except ImportError as e:
    logging.error(f"❌ Critical import error: {e}")
    logging.info("💡 TensorFlow Nightly Fix:")
    logging.info("   export USE_TF=NO")
    logging.info("   pip uninstall tensorflow-nightly tensorflow")
    logging.info("   pip install transformers==4.36.0 torch==2.1.0")
    logging.info("   Restart your Python session")
    sys.exit(1)

# Continue with other imports
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
import psutil
import subprocess

# =========================
# Resource Logging Functions
# =========================
process = psutil.Process(os.getpid())

def log_resource_usage():
    cpu = process.cpu_percent(interval=None)
    mem = process.memory_info().rss / (1024 * 1024)
    logging.info(f"CPU: {cpu:.1f}% | Mem: {mem:.2f} MB")

def log_gpu_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        gpu_stats = result.stdout.strip().split('\n')
        for idx, line in enumerate(gpu_stats):
            util, mem_used, mem_total = map(int, line.split(', '))
            logging.info(f"GPU {idx} - Util: {util}% | Mem: {mem_used} MiB / {mem_total} MiB")
    except Exception as e:
        logging.warning(f"GPU logging failed: {e}")

# =========================
# Environment Check with TF Nightly Detection
# =========================
def check_environment():
    """Check if the environment is properly set up"""
    try:
        import transformers
        import torch
        import peft
        import datasets
        
        logging.info(f"🔍 Environment Check:")
        logging.info(f"   Transformers: {transformers.__version__}")
        logging.info(f"   PyTorch: {torch.__version__}")
        logging.info(f"   PEFT: {peft.__version__}")
        logging.info(f"   Datasets: {datasets.__version__}")
        
        # Check if TensorFlow is being used by transformers
        try:
            tf_available = transformers.utils.is_tf_available()
            logging.info(f"   TF in transformers: {'❌ Disabled' if not tf_available else '⚠️  Enabled'}")
            if tf_available:
                logging.warning("   Consider disabling TF: export USE_TF=NO")
        except:
            logging.info("   TF in transformers: ❌ Disabled")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logging.info(f"   CUDA: {torch.version.cuda}")
            logging.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            logging.warning("   CUDA: Not available")
            
        return True
    except Exception as e:
        logging.error(f"Environment check failed: {e}")
        return False

# =========================
# Dataset Path Check
# =========================
def check_dataset_path():
    """Check if dataset path exists and suggest alternatives"""
    dataset_paths = [
        "Dataset/codealpaca_gsm8k_30k.jsonl",
        "Dataset/categorized_split_data_stratified_train.jsonl",
        "Dataset/Splitted/categorized_split_data_stratified_train.jsonl",
        "Dataset/devstral_inference.jsonl",
        "../../Dataset/categorized_split_data_stratified_train.jsonl",
        "./Dataset/categorized_split_data_stratified_train.jsonl",
        "../Dataset/categorized_split_data_stratified_train.jsonl",
        "/mnt/data/Projects/datathon-2025/Dataset/categorized_split_data_stratified_train.jsonl"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            logging.info(f"✅ Found dataset at: {path}")
            return path
    
    logging.error("❌ Dataset not found in any expected location")
    logging.info("Available files in current directory:")
    for item in os.listdir("."):
        logging.info(f"   {item}")
    return None

# =========================
# Setup Optimizations
# =========================
def setup_torch_optimizations():
    """Setup optimized torch settings for 0.5B model training"""
    # Enable optimized attention (Flash Attention if available)
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Enable memory-efficient attention
    if hasattr(torch.backends.cuda, 'enable_math_sdp'):
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Set memory fraction to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.empty_cache()
    
    logging.info("🔧 Torch optimizations enabled")

# =========================
# FP16 Compatibility Check
# =========================
def check_fp16_compatibility():
    """Check if FP16 is properly supported"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        
        logging.info(f"🎯 GPU: {gpu_name}")
        logging.info(f"   Compute Capability: {compute_capability}")
        
        # Check if Tensor Cores are available (SM 7.0+)
        use_fp16 = compute_capability[0] >= 7
        logging.info(f"   FP16 Tensor Cores: {'✅ Available' if use_fp16 else '❌ Not Available'}")
        
        return use_fp16
    return False

# =========================
# Model Memory Footprint Check
# =========================
def check_model_memory_footprint(model):
    """Optimized memory footprint checking for smaller models"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"🔍 Model Analysis (1.5B optimized):")
    logging.info(f"   Total parameters: {total_params:,}")
    logging.info(f"   Trainable parameters: {trainable_params:,}")
    logging.info(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    # Estimate parameter memory usage
    param_memory_gb = total_params * 4 / (1024**3)  # 4 bytes per param (fp32)
    param_memory_fp16_gb = total_params * 2 / (1024**3)  # 2 bytes per param (fp16)
    
    logging.info(f"   Est. param memory (FP32): {param_memory_gb:.2f} GB")
    logging.info(f"   Est. param memory (FP16): {param_memory_fp16_gb:.2f} GB")
    
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        logging.info(f"   Actual GPU memory: {allocated_memory:.2f} GB")

# =========================
# Environment and Path Checks
# =========================
if not check_environment():
    sys.exit(1)

dataset_path = check_dataset_path()
if not dataset_path:
    logging.error("Cannot proceed without dataset")
    sys.exit(1)

# Setup optimizations
setup_torch_optimizations()
use_fp16 = check_fp16_compatibility()

# =========================
# Tokenization Function
# =========================
def tokenize(batch):
    # Optimized tokenization for 1.5B model
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=384,  # Reduced from 512 for better memory efficiency
        return_tensors="pt"
    )

# =========================
# Load Dataset - UPDATED PATH HANDLING
# =========================
try:
    # Load dataset with proper format handling
    if dataset_path.endswith('.jsonl'):
        # Load JSONL file line by line
        data_lines = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        data_lines.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid JSON line: {line[:50]}...")
                        continue
        
        # Convert to HuggingFace Dataset format
        from datasets import Dataset
        dataset = Dataset.from_list(data_lines)
        
        # Create training text based on available fields
        def format_training_data(example):
            # Handle different dataset formats
            if "instruction" in example and "output" in example:
                # codealpaca_gsm8k format
                text = f"Instruction: {example['instruction']}\nOutput: {example['output']}"
            elif "instruction" in example and "model_output" in example:
                # categorized_split format
                text = f"Instruction: {example['instruction']}\nOutput: {example['model_output']}"
            else:
                # Fallback - use all available text fields
                text_parts = []
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 10:
                        text_parts.append(f"{key}: {value}")
                text = "\n".join(text_parts)
            
            return {"text": text}
        
        dataset = dataset.map(format_training_data)
        logging.info(f"✅ Dataset loaded successfully: {len(dataset)} samples")
        
    else:
        # Fallback to original JSON loading
        dataset = load_dataset("json", data_files={"train": dataset_path})["train"]
        dataset = dataset.map(lambda x: {"text": x["instruction"] + "\n" + x["model_output"]})
        logging.info(f"✅ Dataset loaded successfully: {len(dataset)} samples")
        
except Exception as e:
    logging.error(f"❌ Failed to load dataset: {e}")
    logging.info("💡 Available dataset info:")
    if os.path.exists(dataset_path):
        logging.info(f"   File exists: {dataset_path}")
        logging.info(f"   File size: {os.path.getsize(dataset_path)} bytes")
        # Show first few lines
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 3:
                        logging.info(f"   Line {i+1}: {line[:100]}...")
                    else:
                        break
        except Exception as read_error:
            logging.error(f"   Error reading file: {read_error}")
    sys.exit(1)

# =========================
# Tokenizer - FIXED PATH DETECTION
# =========================
model_paths = [
    "StudentModels/qwen_models/2.5-1.5B/",
    "./StudentModels/qwen_models/2.5-1.5B/",
    "../StudentModels/qwen_models/2.5-1.5B/",
    "/mnt/data/Projects/datathon-2025/StudentModels/qwen_models/2.5-1.5B/"
]

model_dir = None
for path in model_paths:
    if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
        model_dir = path
        break

if not model_dir:
    logging.error("❌ Model directory not found in any expected location")
    logging.info("💡 Run the model verification tool first:")
    logging.info("   python Prep/verify-and-download-qwen-1.5b.py")
    sys.exit(1)

logging.info(f"✅ Using model from: {model_dir}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    logging.info("✅ Tokenizer loaded successfully")
except Exception as e:
    logging.error(f"❌ Failed to load tokenizer: {e}")
    sys.exit(1)

# Apply tokenization to dataset
try:
    dataset = dataset.map(tokenize, batched=True, batch_size=1000)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    logging.info(f"✅ Dataset tokenized successfully")
except Exception as e:
    logging.error(f"❌ Failed to tokenize dataset: {e}")
    sys.exit(1)

# =========================
# Model Loading with Enhanced Error Handling
# =========================
try:
    logging.info("🔧 Setting up optimized quantization config for 1.5B...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16 if use_fp16 else torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.uint8
    )

    logging.info("📥 Loading 1.5B model with optimizations...")
    log_gpu_usage()

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_fp16 else torch.bfloat16,
        low_cpu_mem_usage=True,
        use_cache=False  # Disable KV cache during training to save memory
    )
    
    logging.info("✅ 1.5B model loaded successfully")
    
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    logging.info("💡 Troubleshooting suggestions:")
    logging.info("   1. Check GPU memory: nvidia-smi")
    logging.info("   2. Verify model files are complete")
    logging.info("   3. Try reducing quantization settings")
    sys.exit(1)

# =========================
# Optimized LoRA Configuration
# =========================
logging.info("🔧 Applying optimized LoRA adapter...")
lora_config = LoraConfig(
    r=8,               # Reduced rank for 1.5B model
    lora_alpha=16,     # Proportional alpha
    lora_dropout=0.1,  # Slightly higher dropout
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # More comprehensive targeting
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

logging.info("✅ LoRA applied, checking final memory footprint...")
log_gpu_usage()
check_model_memory_footprint(model)

model.print_trainable_parameters()

# =========================
# Training Arguments - OPTIMIZED FOR 1.5B
# =========================
training_args = TrainingArguments(
    output_dir="StudentModels/Adapters/qwen2.5-1.5b-lora/",
    per_device_train_batch_size=2,      # Optimized batch size for 1.5B
    gradient_accumulation_steps=8,      # Balanced accumulation
    num_train_epochs=2,                 # More epochs for better training
    learning_rate=1e-4,                 # Optimized learning rate
    fp16=use_fp16,                      # Dynamic FP16 based on hardware
    bf16=not use_fp16,                  # Use bf16 if fp16 not available
    logging_steps=25,                   # More frequent logging
    save_strategy="steps",
    save_steps=250,                     # More frequent saves
    dataloader_num_workers=2,           # Optimized workers for 1.5B
    dataloader_pin_memory=True,         # Pin memory for faster transfers
    remove_unused_columns=False,
    report_to="none",
    max_grad_norm=0.5,                  # Tighter gradient clipping
    warmup_steps=50,                    # Reduced warmup
    weight_decay=0.05,                  # Increased regularization
    lr_scheduler_type="cosine",         # Cosine annealing
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",  # Fused optimizer
    group_by_length=True,               # Group similar lengths for efficiency
    dataloader_drop_last=True           # Drop incomplete batches
)

# =========================
# Enhanced Resource Monitor Callback
# =========================
losses = []

class OptimizedResourceCallback(TrainerCallback):
    def __init__(self):
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            loss_val = logs["loss"]
            losses.append(loss_val)
            
            # Track best loss
            if loss_val < self.best_loss:
                self.best_loss = loss_val
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1
            
            logging.info(f"Step {state.global_step} | Loss: {loss_val:.4f} | Best: {self.best_loss:.4f}")
        
        # Log resources less frequently to reduce overhead
        if state.global_step % 25 == 0:
            log_resource_usage()
            log_gpu_usage()
    
    def on_step_end(self, args, state, control, **kwargs):
        # Memory cleanup every 100 steps
        if state.global_step % 100 == 0:
            torch.cuda.empty_cache()

# =========================
# Optimized Trainer Setup
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8  # Optimize for tensor cores
    ),
    callbacks=[OptimizedResourceCallback()]
)

# =========================
# Pre-training Memory Optimization
# =========================
logging.info("🚀 Final optimizations before training:")

# Clear any cached memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Final memory check
log_gpu_usage()

if torch.cuda.is_available():
    allocated_memory = torch.cuda.memory_allocated() / 1024**3
    reserved_memory = torch.cuda.memory_reserved() / 1024**3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    logging.info(f"📊 Pre-training Memory Status:")
    logging.info(f"   Allocated: {allocated_memory:.2f} GB")
    logging.info(f"   Reserved: {reserved_memory:.2f} GB") 
    logging.info(f"   Total GPU: {total_memory:.2f} GB")
    logging.info(f"   Usage: {(allocated_memory/total_memory)*100:.1f}%")
    
    # Memory efficiency check
    if allocated_memory > total_memory * 0.8:
        logging.warning(f"⚠️  HIGH MEMORY USAGE: {allocated_memory:.2f} GB!")
        logging.warning("Consider reducing batch size or sequence length")

# =========================
# Start Optimized Training
# =========================
try:
    logging.info("🎯 Starting optimized 1.5B training...")
    trainer.train()

    logging.info("💾 Saving optimized model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    logging.info("✅ Training completed successfully!")
    
except KeyboardInterrupt:
    logging.info("⚠️  Training interrupted by user")
    # Save checkpoint if possible
    try:
        trainer.save_model(os.path.join(training_args.output_dir, "interrupted_checkpoint"))
        logging.info("💾 Saved interrupted checkpoint")
    except:
        pass
        
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        logging.error("❌ GPU Out of Memory!")
        logging.info("💡 Try these solutions:")
        logging.info("   1. Reduce batch size: per_device_train_batch_size=1")
        logging.info("   2. Increase gradient accumulation: gradient_accumulation_steps=16")
        logging.info("   3. Reduce sequence length: max_length=256")
        logging.info("   4. Use gradient checkpointing")
    else:
        logging.error(f"❌ Runtime error: {e}")
    sys.exit(1)
    
except Exception as e:
    logging.error(f"❌ Unexpected error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # =========================
    # Enhanced Loss Visualization
    # =========================
    if losses:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(losses, label='Training Loss', linewidth=2)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Qwen 1.5B Optimized LoRA Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add moving average
        if len(losses) > 10:
            window_size = min(50, len(losses) // 10)
            moving_avg = [sum(losses[max(0, i-window_size):i+1])/min(i+1, window_size) for i in range(len(losses))]
            plt.subplot(2, 1, 2)
            plt.plot(moving_avg, label=f'Moving Average (window={window_size})', color='orange', linewidth=2)
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Smoothed Training Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qwen_15_optimized_training_loss.png', dpi=300, bbox_inches='tight')
        logging.info("📊 Saved enhanced loss plot to qwen_15_optimized_training_loss.png")

    logging.info("✅ Optimized 1.5B training completed!")
    log_gpu_usage()
