# Setup Guide for Property Recommendation System

This guide explains how to set up and run the Property Recommendation System using the provided scripts.

## Requirements

- Python 3.x
- pip (Python package manager)
- NVIDIA GPU with CUDA support (optional, for better performance)
- NVIDIA CUDA drivers installed on your system

## Data Requirements

The system requires a JSON dataset file at `data/appraisals_dataset.json`. The setup script will:

- Check if this file exists
- Offer to create a sample dataset file if it doesn't exist (for testing purposes)
- Allow you to continue without it (but the application will likely fail)

## Setup and Run Instructions

### For Windows Users

1. Open Command Prompt or PowerShell in the project directory
2. Run the batch script:
   ```
   run.bat
   ```

The script will:

- Check if Python and pip are installed
- Create a virtual environment (if it doesn't exist)
- Prompt you to select a CUDA version for PyTorch:
  - CUDA 11.8 (recommended for most GPUs)
  - CUDA 12.1 (for newest GPUs)
  - CPU-only fallback
- Install PyTorch with your selected CUDA version
- Attempt to install triton (optional package, will continue if it fails)
- Install unsloth and configure it to work without triton if necessary
- Install tf-keras (required for compatibility with Transformers and Keras 3)
- Patch source files to handle import issues
- Set up necessary CUDA environment variables
- Check for required data files and offer to create a sample dataset
- Run the main application
- Close the virtual environment when done

### For Linux/Mac Users

1. Open Terminal in the project directory
2. Make the script executable (first time only):
   ```
   chmod +x run.sh
   ```
3. Run the bash script:
   ```
   ./run.sh
   ```

The script will perform the same steps as the Windows batch script.

## CUDA Version Selection

When running the script, you'll be prompted to select a CUDA version:

1. **CUDA 11.8** - This is the recommended option for most GPUs, with better compatibility across different NVIDIA GPUs, especially for slightly older models.

2. **CUDA 12.1** - This is the latest version, which may offer better performance for the newest NVIDIA GPUs (RTX 40 series and newer).

3. **CPU-only** - Fallback option if you encounter issues with the CUDA versions.

If you're unsure which CUDA version to choose, start with CUDA 11.8, as it has broader compatibility.

## Manual Setup

If the scripts don't work for your system, you can set up the project manually:

1. Create a virtual environment:

   ```
   python -m venv venv
   ```

2. Activate the virtual environment:

   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install PyTorch with CUDA (choose one):

   - For CUDA 11.8:
     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - For CUDA 12.1:
     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - For CPU-only:
     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```

4. Verify CUDA setup:

   ```
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
   ```

5. Try to install triton (optional):

   ```
   pip install triton --no-deps
   ```

   Note: This package may not be available for all systems. The system will work without it.

6. Install unsloth and its dependencies:

   ```
   pip install unsloth --no-deps
   pip install accelerate bitsandbytes peft transformers trl
   ```

7. Install remaining requirements:

   ```
   pip install -r requirements.txt
   ```

8. Install tf-keras (required for compatibility with Transformers):

   ```
   pip install tf-keras
   ```

9. Create or verify the data file:

   - Make sure you have a `data` directory in the project root
   - Create a file named `data/appraisals_dataset.json` with your dataset
   - For testing, you can create a minimal dataset:
     ```json
     [
       {
         "property_id": "sample1",
         "location": { "latitude": 40.7128, "longitude": -74.006 },
         "features": {
           "bedrooms": 3,
           "bathrooms": 2,
           "area": 1500,
           "year_built": 2010
         },
         "price": 350000
       }
     ]
     ```

10. Set CUDA environment variables:

    - Windows: `set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
    - Linux/Mac: `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

11. Run the application:

    ```
    python src/main.py
    ```

12. Deactivate the virtual environment when done:
    ```
    deactivate
    ```

## Troubleshooting

If you encounter any issues:

1. Ensure Python 3.x is installed and in your PATH
2. Check that pip is installed and updated
3. Make sure you're running the scripts from the project root directory
4. If you encounter a Keras compatibility error, make sure tf-keras is installed:
   ```
   pip install tf-keras
   ```
5. If you encounter a "No module named 'triton'" error:
   - This is normal as triton is an optional package not available on all systems
   - The scripts have been updated to handle this case automatically
   - You can continue using the system without triton
6. If you encounter a "FileNotFoundError" for the appraisals_dataset.json file:
   - Create the data directory: `mkdir -p data`
   - Create a sample dataset file as shown in step 9 of the manual setup
7. If you encounter CUDA/GPU errors:
   - Make sure your NVIDIA drivers are up to date (check in NVIDIA Control Panel)
   - Verify your GPU is compatible with the CUDA version you selected
   - Try the other CUDA version (11.8 or 12.1)
   - Use `nvidia-smi` command to check if your GPU is properly detected
   - Set memory allocation limits: `set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
   - If problems persist, fall back to the CPU version:
     ```
     pip uninstall -y torch torchvision torchaudio
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```
