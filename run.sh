#!/bin/bash

# Define colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Property Recommendation System - Setup and Run Script${NC}"
echo -e "${YELLOW}Checking requirements...${NC}"

# Check if Python is installed
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python is not installed. Please install Python 3.x and try again.${NC}"
    exit 1
fi

# Determine python command (python or python3)
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo -e "${RED}pip is not installed. Please install pip and try again.${NC}"
    exit 1
fi

# Determine pip command (pip or pip3)
PIP_CMD="pip"
if ! command -v pip &> /dev/null; then
    PIP_CMD="pip3"
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}requirements.txt not found. Make sure you're running this script from the project root directory.${NC}"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please install venv and try again.${NC}"
        echo -e "${YELLOW}You can install venv with: $PIP_CMD install virtualenv${NC}"
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment. Please check your Python installation.${NC}"
    exit 1
fi

echo -e "${GREEN}Virtual environment activated.${NC}"

# Install or upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
$PIP_CMD install --upgrade pip

# Uninstall torch to reinstall the proper version
echo -e "${YELLOW}Uninstalling existing torch installation...${NC}"
$PIP_CMD uninstall -y torch torchvision torchaudio

# Show CUDA installation options
echo -e "${YELLOW}You indicated that you have CUDA installed and working on your system.${NC}"
echo -e "Which CUDA version would you like to use with PyTorch?"
echo -e "1. CUDA 11.8 (recommended for most newer GPUs)"
echo -e "2. CUDA 12.1 (latest, for newest GPUs)"
echo -e "3. CPU-only version (fallback)"
read -p "Enter choice (1, 2, or 3): " CUDA_CHOICE

# Install PyTorch with the appropriate CUDA version
if [ "$CUDA_CHOICE" = "1" ]; then
    echo -e "${YELLOW}Installing PyTorch with CUDA 11.8 support...${NC}"
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "$CUDA_CHOICE" = "2" ]; then
    echo -e "${YELLOW}Installing PyTorch with CUDA 12.1 support...${NC}"
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo -e "${YELLOW}Installing CPU-only version of PyTorch...${NC}"
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify CUDA setup
echo -e "${YELLOW}Verifying CUDA setup...${NC}"
$PYTHON_CMD -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A')"

# Try to install triton for unsloth, but continue if it fails
echo -e "${YELLOW}Attempting to install triton (optional)...${NC}"
$PIP_CMD install triton --no-deps 2>/dev/null || echo -e "${YELLOW}Triton installation failed. This is expected on some systems and can be ignored.${NC}"

# Install unsloth with CUDA support (BEFORE other requirements)
echo -e "${YELLOW}Installing unsloth with CUDA support...${NC}"
$PIP_CMD uninstall -y unsloth
$PIP_CMD install unsloth --no-deps

# Modify unsloth to work without triton if needed
echo -e "${YELLOW}Setting up unsloth to work without triton if necessary...${NC}"
UNSLOTH_INIT_FILE="venv/lib/python*/site-packages/unsloth/__init__.py"
UNSLOTH_INIT_PATH=$(ls $UNSLOTH_INIT_FILE 2>/dev/null || echo "")

if [ -n "$UNSLOTH_INIT_PATH" ]; then
    $PYTHON_CMD -c "
path = '$UNSLOTH_INIT_PATH'
try:
    with open(path, 'r') as f:
        content = f.read()
    with open(path, 'w') as f:
        f.write(content.replace('import triton', 'try:\\n    import triton\\nexcept ImportError:\\n    print(\"Triton not available, some optimizations disabled\")'))
    print('Modified unsloth __init__.py to handle missing triton')
except:
    print('Could not modify unsloth __init__.py')
"
fi

# Install required dependencies
echo -e "${YELLOW}Installing accelerate bitsandbytes peft transformers trl...${NC}"
$PIP_CMD install accelerate bitsandbytes peft transformers trl

# Install tf-keras for compatibility with Transformers
echo -e "${YELLOW}Installing tf-keras for compatibility with Transformers...${NC}"
$PIP_CMD install tf-keras

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install tf-keras. This might cause issues with Transformers.${NC}"
fi

# Install requirements
echo -e "${YELLOW}Installing remaining requirements...${NC}"
$PIP_CMD install -r requirements.txt

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install requirements. Please check your requirements.txt file.${NC}"
    exit 1
fi

echo -e "${GREEN}All requirements installed successfully!${NC}"

# Create environment variables for CUDA
echo -e "${YELLOW}Setting up CUDA environment variables...${NC}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Create a temporary fix for the import order issue
echo -e "${YELLOW}Fixing import order issue in main.py...${NC}"
MAIN_FILE="src/main.py"
TEMP_FILE="src/main_fixed.py"

cp "$MAIN_FILE" "$TEMP_FILE"

# Run Python to modify the main_fixed.py file to fix import order
$PYTHON_CMD -c "
with open('$TEMP_FILE', 'r') as f:
    content = f.read()
content = 'try:\\n    import unsloth\\nexcept ImportError:\\n    print(\"Unsloth import failed, some functionality may be limited\")\\n' + content
with open('$TEMP_FILE', 'w') as f:
    f.write(content)
"

# Update the llama_recommender.py file if it exists to handle missing triton
LLAMA_FILE="src/llama_recommender.py"
if [ -f "$LLAMA_FILE" ]; then
    LLAMA_BACKUP="src/llama_recommender.py.bak"
    cp "$LLAMA_FILE" "$LLAMA_BACKUP"
    
    $PYTHON_CMD -c "
with open('$LLAMA_FILE', 'r') as f:
    content = f.read()
content = content.replace('from unsloth import FastLanguageModel', 'try:\\n    from unsloth import FastLanguageModel\\nexcept ImportError:\\n    print(\"FastLanguageModel import failed, functionality will be limited\")')
with open('$LLAMA_FILE', 'w') as f:
    f.write(content)
"
fi

# Run the fixed main.py file
echo -e "${YELLOW}Running main_fixed.py...${NC}"
$PYTHON_CMD src/main_fixed.py

# Deactivate virtual environment
deactivate

echo -e "${GREEN}Done!${NC}" 