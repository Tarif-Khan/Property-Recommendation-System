#!/usr/bin/env python
"""
Setup script for installing and validating LLAMA model for the Property Recommendation System
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required dependencies for LLAMA model fine-tuning."""
    logger.info("Installing dependencies for LLAMA fine-tuning...")
    
    # Required packages
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "unsloth>=2023.12",
        "bitsandbytes>=0.41.0",
        "datasets>=2.14.0",
        "huggingface_hub>=0.19.0",
        "peft>=0.6.0"
    ]
    
    # Install each requirement
    for req in requirements:
        logger.info(f"Installing {req}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {req}: {str(e)}")
            return False
    
    logger.info("Dependencies installed successfully!")
    return True

def download_model(model_name, cache_dir=None):
    """Download the LLAMA model and tokenizer."""
    logger.info(f"Downloading model: {model_name}")
    
    try:
        # Import the necessary libraries
        from transformers import AutoTokenizer
        from unsloth import FastLanguageModel
        import torch
        
        # Create cache directory if specified
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Using cache directory: {cache_dir}")
        
        # Download the tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Download the model
        logger.info("Downloading model weights (this may take a while)...")
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            cache_dir=cache_dir
        )
        
        logger.info("Model and tokenizer downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

def test_model(model_name, cache_dir=None):
    """Test the model by running a simple inference."""
    logger.info("Testing model with a simple inference...")
    
    try:
        # Import the necessary libraries
        from transformers import AutoTokenizer
        from unsloth import FastLanguageModel
        import torch
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            cache_dir=cache_dir
        )
        
        # Set padding token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare a simple test input
        test_prompt = "<s>[INST] What makes a property valuable? [/INST]"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = model.to(device)
        
        # Run inference
        logger.info("Running inference (this may take a few seconds)...")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True
            )
            
        # Decode output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if output makes sense
        logger.info(f"Model test output: {output_text}")
        logger.info("Model test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup LLAMA model for Property Recommendation System")
    
    parser.add_argument("--model", type=str, default="unsloth/llama-3-8b-bnb-4bit",
                        help="LLAMA model to use (default: unsloth/llama-3-8b-bnb-4bit)")
    parser.add_argument("--cache_dir", type=str, default="cache/llama",
                        help="Directory to cache the model (default: cache/llama)")
    parser.add_argument("--skip_deps", action="store_true",
                        help="Skip dependency installation")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip model testing")
    
    args = parser.parse_args()
    
    logger.info("LLAMA Model Setup for Property Recommendation System")
    logger.info("==================================================")
    
    # Install dependencies if not skipped
    if not args.skip_deps:
        if not install_dependencies():
            logger.error("Failed to install dependencies. Exiting...")
            return 1
    else:
        logger.info("Skipping dependency installation.")
    
    # Download model
    if not download_model(args.model, args.cache_dir):
        logger.error("Failed to download the model. Exiting...")
        return 1
    
    # Test model if not skipped
    if not args.skip_test:
        if not test_model(args.model, args.cache_dir):
            logger.error("Model test failed. The model may not work properly.")
            return 1
    else:
        logger.info("Skipping model testing.")
    
    logger.info("==================================================")
    logger.info("LLAMA model setup completed successfully!")
    logger.info(f"Model: {args.model}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info("")
    logger.info("You can now use the LLAMA-based recommender with the following commands:")
    logger.info(f"  python src/main.py --method llama --llama_model {args.model} --llama_cache_dir {args.cache_dir}")
    logger.info("  OR")
    logger.info(f"  streamlit run src/app.py  # Select LLAMA in the web interface")
    logger.info("==================================================")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 