#!/bin/bash

# Enhanced script to merge all student models with their adapters
# This script provides multiple options for merging models

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if we're in the right directory
check_directory() {
    if [[ ! -d "StudentModels" ]]; then
        error "StudentModels directory not found!"
        error "Please run this script from the project root directory"
        exit 1
    fi
    
    if [[ ! -d "Scripts" ]]; then
        error "Scripts directory not found!"
        error "Please run this script from the project root directory"
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    log "Checking Python dependencies..."
    
    python3 -c "import transformers, peft, torch" 2>/dev/null || {
        error "Required Python packages not found!"
        echo "Please install required packages:"
        echo "  pip install transformers peft torch"
        exit 1
    }
    
    success "All dependencies found"
}

# Display available models
show_available_models() {
    log "Scanning for available models..."
    echo
    echo "📋 Available Base Models:"
    if [[ -d "StudentModels/qwen_models" ]]; then
        for model in StudentModels/qwen_models/*/; do
            if [[ -d "$model" ]]; then
                basename_model=$(basename "$model")
                echo "  ✅ $basename_model"
            fi
        done
    else
        warning "No base models found in StudentModels/qwen_models/"
    fi
    
    echo
    echo "📋 Available Adapters:"
    if [[ -d "StudentModels/Adapters" ]]; then
        for adapter in StudentModels/Adapters/*/; do
            if [[ -d "$adapter" ]]; then
                basename_adapter=$(basename "$adapter")
                echo "  ✅ $basename_adapter"
                
                # Show checkpoints if available
                if ls "$adapter"checkpoint-*/ >/dev/null 2>&1; then
                    echo "    📁 Checkpoints:"
                    for checkpoint in "$adapter"checkpoint-*/; do
                        if [[ -d "$checkpoint" ]]; then
                            checkpoint_name=$(basename "$checkpoint")
                            echo "      - $checkpoint_name"
                        fi
                    done
                fi
            fi
        done
    else
        warning "No adapters found in StudentModels/Adapters/"
    fi
}

# Merge all models automatically
merge_all_models() {
    log "Starting automatic merge of all models..."
    
    # Create merged models directory
    mkdir -p StudentModels/MergedModels
    
    # Run the enhanced Python script
    python3 Scripts/merge-model.py --merge_all
    
    if [[ $? -eq 0 ]]; then
        success "All models merged successfully!"
        log "Merged models are available in: StudentModels/MergedModels/"
    else
        error "Some models failed to merge. Check the logs for details."
        exit 1
    fi
}

# Interactive merge
interactive_merge() {
    log "Starting interactive merge..."
    
    echo "Available merge options:"
    echo "1. Merge specific model pair"
    echo "2. Merge all models automatically"
    echo "3. Show model information only"
    echo
    
    read -p "Choose option (1-3): " choice
    
    case $choice in
        1)
            echo "Feature coming soon - use automatic merge for now"
            merge_all_models
            ;;
        2)
            merge_all_models
            ;;
        3)
            show_available_models
            ;;
        *)
            error "Invalid choice"
            exit 1
            ;;
    esac
}

# Test merged models
test_merged_models() {
    log "Testing merged models..."
    
    if [[ ! -d "StudentModels/MergedModels" ]]; then
        warning "No merged models found. Run merge first."
        return 1
    fi
    
    for merged_model in StudentModels/MergedModels/*/; do
        if [[ -d "$merged_model" ]]; then
            model_name=$(basename "$merged_model")
            log "Testing $model_name..."
            
            # Check if required files exist
            if [[ -f "$merged_model/config.json" && -f "$merged_model/tokenizer.json" ]]; then
                success "$model_name appears to be properly merged"
            else
                warning "$model_name may be incomplete"
            fi
        fi
    done
}

# Main menu
main() {
    echo "🤖 SemiQwenn Model Merger"
    echo "=========================="
    echo
    
    # Check prerequisites
    check_directory
    check_dependencies
    
    # Parse command line arguments
    case "${1:-interactive}" in
        "all"|"--all")
            merge_all_models
            ;;
        "show"|"--show")
            show_available_models
            ;;
        "test"|"--test")
            test_merged_models
            ;;
        "interactive"|"")
            interactive_merge
            ;;
        "help"|"--help"|"-h")
            echo "Usage: $0 [option]"
            echo
            echo "Options:"
            echo "  all         Merge all available models automatically"
            echo "  show        Show available models and adapters"
            echo "  test        Test merged models"
            echo "  interactive Interactive mode (default)"
            echo "  help        Show this help message"
            echo
            echo "Examples:"
            echo "  $0 all              # Merge all models"
            echo "  $0 show             # Show available models"
            echo "  $0                  # Interactive mode"
            ;;
        *)
            error "Unknown option: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
