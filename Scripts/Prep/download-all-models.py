"""
Master script to download all models (teacher and student) for the datathon project.
"""

import sys
from pathlib import Path

# Add current directory to path to import other download scripts
sys.path.append(str(Path(__file__).parent))

from download_teacher_models import download_teacher_models
from download_qwen_7b import download_qwen_7b
from download_qwen_1_5b import download_qwen_1_5b
from download_qwen_0_5b import download_qwen_0_5b

def main():
    """Download all models in sequence."""
    print("🚀 Starting download of all models for datathon-2025 project\n")
    
    results = {}
    
    # Download teacher models
    print("Step 1: Downloading teacher models...")
    results['teacher'] = download_teacher_models()
    
    # Download student models
    print("\nStep 2: Downloading student models...")
    
    print("\n2a. Downloading Qwen2.5-7B...")
    results['qwen_7b'] = download_qwen_7b()
    
    print("\n2b. Downloading Qwen2.5-1.5B...")
    results['qwen_1_5b'] = download_qwen_1_5b()
    
    print("\n2c. Downloading Qwen2.5-0.5B...")
    results['qwen_0_5b'] = download_qwen_0_5b()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL DOWNLOAD SUMMARY")
    print("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for model_type, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{model_type:<15}: {status}")
    
    print("-" * 60)
    print(f"Overall: {success_count}/{total_count} model groups downloaded successfully")
    
    if success_count == total_count:
        print("🎉 All models downloaded successfully! Ready for datathon.")
    else:
        print("⚠️  Some downloads failed. Check individual logs for details.")
    
    return success_count == total_count

if __name__ == "__main__":
    main()
