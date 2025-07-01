import subprocess
import sys
import time
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run_all_base_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run base model inference"""
    logger.info("🚀 Starting base model inference for all student models")
    start_time = time.time()
    
    try:
        # Change to project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        os.chdir(project_root)
        
        # Run the inference script from the correct location
        script_path = os.path.join("Scripts", "Inference", "infer_base_models.py")
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Base model inference completed in {total_time:.2f} seconds")
        logger.info("📋 Final output:")
        logger.info(result.stdout[-1000:])  # Last 1000 chars
        
        return True
        
    except subprocess.CalledProcessError as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Base model inference failed after {total_time:.2f} seconds")
        logger.error(f"Error: {e.stderr}")
        logger.error(f"Output: {e.stdout}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 All base models inference completed successfully!")
    else:
        logger.error("❌ Base model inference failed")
        sys.exit(1)
