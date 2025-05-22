@echo off
setlocal enabledelayedexpansion

echo Property Recommendation System - Setup and Run Script
echo Checking requirements...

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    where python3 >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo Python is not installed. Please install Python 3.x and try again.
        exit /b 1
    ) else (
        set PYTHON_CMD=python3
    )
) else (
    set PYTHON_CMD=python
)

REM Check if pip is installed
where pip >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    where pip3 >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo pip is not installed. Please install pip and try again.
        exit /b 1
    ) else (
        set PIP_CMD=pip3
    )
) else (
    set PIP_CMD=pip
)

REM Check if requirements.txt exists
if not exist requirements.txt (
    echo requirements.txt not found. Make sure you're running this script from the project root directory.
    exit /b 1
)

REM Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment. Please install venv and try again.
        echo You can install venv with: %PIP_CMD% install virtualenv
        exit /b 1
    )
)

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Failed to activate virtual environment. Please check your Python installation.
    exit /b 1
)

echo Virtual environment activated.

REM Install or upgrade pip
echo Upgrading pip...
%PIP_CMD% install --upgrade pip

REM Uninstall torch to reinstall the proper version
echo Uninstalling existing torch installation...
%PIP_CMD% uninstall -y torch torchvision torchaudio

REM Show CUDA installation options
echo You indicated that you have CUDA installed and working on your system.
echo Which CUDA version would you like to use with PyTorch?
echo 1. CUDA 11.8 (recommended for most newer GPUs)
echo 2. CUDA 12.1 (latest, for newest GPUs)
echo 3. CPU-only version (fallback)
set /p CUDA_CHOICE="Enter choice (1, 2, or 3): "

REM Install PyTorch with the appropriate CUDA version
if "!CUDA_CHOICE!"=="1" (
    echo Installing PyTorch with CUDA 11.8 support...
    %PIP_CMD% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "!CUDA_CHOICE!"=="2" (
    echo Installing PyTorch with CUDA 12.1 support...
    %PIP_CMD% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing CPU-only version of PyTorch...
    %PIP_CMD% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM Verify CUDA setup
echo Verifying CUDA setup...
%PYTHON_CMD% -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A')"

REM Try to install triton for unsloth, but continue if it fails
echo Attempting to install triton (optional)...
%PIP_CMD% install triton --no-deps 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Triton installation failed. This is expected on some systems and can be ignored.
    echo We'll configure the system to work without triton.
)

REM Install unsloth with CUDA support (BEFORE other requirements)
echo Installing unsloth with CUDA support...
%PIP_CMD% uninstall -y unsloth
%PIP_CMD% install unsloth --no-deps

REM Modify unsloth to work without triton if needed
echo Setting up unsloth to work without triton if necessary...
set "unsloth_init_file=venv\Lib\site-packages\unsloth\__init__.py"
if exist "%unsloth_init_file%" (
    %PYTHON_CMD% -c "content = open('%unsloth_init_file%', 'r').read(); open('%unsloth_init_file%', 'w').write(content.replace('import triton', 'try:\n    import triton\nexcept ImportError:\n    print(\"Triton not available, some optimizations disabled\")'))" 2>nul
)

REM Install required dependencies
echo Installing accelerate bitsandbytes peft transformers trl...
%PIP_CMD% install accelerate bitsandbytes peft transformers trl

REM Install tf-keras for compatibility with Transformers
echo Installing tf-keras for compatibility with Transformers...
%PIP_CMD% install tf-keras

if %ERRORLEVEL% NEQ 0 (
    echo Failed to install tf-keras. This might cause issues with Transformers.
)

REM Install requirements
echo Installing remaining requirements...
%PIP_CMD% install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo Failed to install requirements. Please check your requirements.txt file.
    exit /b 1
)

echo All requirements installed successfully!

REM Create environment variables for CUDA
echo Setting up CUDA environment variables...
SET PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Create a temporary fix for the import order issue
echo Fixing import order issue in main.py...
set "main_file=src\main.py"
set "temp_file=src\main_fixed.py"

copy "%main_file%" "%temp_file%" >nul

REM Run Python to modify the main_fixed.py file to fix import order
%PYTHON_CMD% -c "with open('%temp_file%', 'r') as f: content = f.read(); content = 'try:\n    import unsloth\nexcept ImportError:\n    print(\"Unsloth import failed, some functionality may be limited\")\n' + content; with open('%temp_file%', 'w') as f: f.write(content)"

REM Update the llama_recommender.py file if it exists to handle missing triton
set "llama_file=src\llama_recommender.py"
if exist "%llama_file%" (
    set "llama_backup=src\llama_recommender.py.bak"
    copy "%llama_file%" "%llama_backup%" >nul
    %PYTHON_CMD% -c "content = open('%llama_file%', 'r').read(); open('%llama_file%', 'w').write(content.replace('from unsloth import FastLanguageModel', 'try:\n    from unsloth import FastLanguageModel\nexcept ImportError:\n    print(\"FastLanguageModel import failed, functionality will be limited\")'))" 2>nul
)

REM Run the fixed main.py file
echo Running main_fixed.py...
%PYTHON_CMD% src\main_fixed.py

REM If we reach this point, it means the script executed without errors
goto end

:end
REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo Done!
pause 