import subprocess
import sys
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run_all_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_model_script(script_name):
    """Run a model inference script"""
    logger.info(f"🚀 Starting {script_name}")
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        total_time = time.time() - start_time
        logger.info(f"✅ {script_name} completed in {total_time:.2f} seconds")
        logger.info(f"Output: {result.stdout[-500:]}")  # Last 500 chars
        
        return True
        
    except subprocess.CalledProcessError as e:
        total_time = time.time() - start_time
        logger.error(f"❌ {script_name} failed after {total_time:.2f} seconds")
        logger.error(f"Error: {e.stderr}")
        return False

def main():
    """Run all model inference scripts"""
    scripts = [
        "infer_qwen_0_5b.py",  # Start with smallest model
        "infer_qwen_1_5b.py",  # Medium model
        "infer_qwen_7b.py"     # Largest model last
    ]
    
    logger.info("🎯 Starting inference for all models")
    start_time = time.time()
    
    results = {}
    
    for script in scripts:
        success = run_model_script(script)
        results[script] = success
        
        if not success:
            logger.error(f"❌ Stopping due to failure in {script}")
            break
        
        # Brief pause between models for cleanup
        time.sleep(5)
    
    total_time = time.time() - start_time
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 INFERENCE SUMMARY")
    logger.info("=" * 60)
    
    for script, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{script}: {status}")
    
    successful = sum(results.values())
    total = len(results)
    
    logger.info(f"🎯 Total: {successful}/{total} models completed successfully")
    logger.info(f"⏱️  Total runtime: {total_time:.2f} seconds")
    
    if successful == total:
        logger.info("🎉 All models completed successfully!")
    else:
        logger.error("⚠️  Some models failed to complete")

if __name__ == "__main__":
    main()
