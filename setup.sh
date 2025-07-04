#!/bin/bash
# Setup script for CDRC framework

echo "============================================"
echo "CDRC Framework Setup Script"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
        exit 1
    fi
}

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_status 0 "Python version OK ($python_version)"
else
    print_status 1 "Python version must be >= 3.8 (found $python_version)"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "cdrc_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv cdrc_env
    print_status $? "Virtual environment created"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source cdrc_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
print_status $? "pip upgraded"

# Install PyTorch (CUDA 11.8 version - adjust as needed)
echo "Installing PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
print_status $? "PyTorch installed"

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
print_status $? "Requirements installed"

# Create directory structure
echo "Creating directory structure..."
mkdir -p cache
mkdir -p processed_data
mkdir -p biased_models
mkdir -p results_llama2_7B_Real/test_toxic_new
mkdir -p results_llama2_7B_Real/conformal_set
mkdir -p results_detoxify_0.15
mkdir -p var_results_detoxify_0.15
mkdir -p figures
mkdir -p tables
mkdir -p jigsaw_data
mkdir -p cdrc_experiments
apt install swig
# Install crossprob from source
# echo "Installing crossprob from source..."
# if ! python -c "import crossprob" 2>/dev/null; then
#     # Install FFTW3 if not present
#     if ! ldconfig -p | grep -q libfftw3; then
#         echo "FFTW3 not found. Please install it:"
#         echo "  Ubuntu/Debian: sudo apt-get install libfftw3-dev"
#         echo "  CentOS/RHEL: sudo yum install fftw-devel"
#         echo "  MacOS: brew install fftw"
#         echo ""
#         echo "After installing FFTW3, run this script again."
#         exit 1
#     fi
    
#     # Clone and build crossing-probability
#     git clone https://github.com/mosco/crossing-probability.git temp_crossprob
#     cd temp_crossprob
    
#     # Build the C++ library
#     make
    
#     # Build and install Python extension
#     make python
#     python setup.py install
    
#     cd ..
    
#     # Verify installation
#     if python -c "import crossprob" 2>/dev/null; then
#         echo "✓ crossprob installed successfully"
#         rm -rf temp_crossprob
#     else
#         echo "✗ crossprob installation failed"
#         echo "Please check the error messages above"
#         exit 1
#     fi
# else
#     echo "✓ crossprob already installed"
# fi

# Create directory structure
echo "Creating directory structure..."
directories=(
    "cache"
    "processed_data"
    "biased_models"
    "results_detoxify_0.15/DRC"
    "results_detoxify_0.15/DKW"
    "results_detoxify_0.15/BJ"
    "var_results_detoxify_0.15/DRC"
    "var_results_detoxify_0.15/DKW"
    "var_results_detoxify_0.15/BJ"
    "results_llama2_7B_Real/test_toxic_new"
    "results_llama2_7B_Real/conformal_set"
    "figures"
    "tables"
    "logs"
    "jigsaw_data"
    "cdrc_experiments/conformal_sets"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done
print_status $? "Directory structure created"

# Check for Jigsaw data
echo "Checking for Jigsaw dataset..."
if [ -f "jigsaw_data/train.csv" ]; then
    print_status 0 "Jigsaw dataset found"
else
    echo -e "${YELLOW}Warning: Jigsaw dataset not found!${NC}"
    echo "Please download train.csv from:"
    echo "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"
    echo "and place it in ./jigsaw_data/"
fi

# Set environment variables
echo "Setting environment variables..."
export HF_HOME="./cache/huggingface"
export TRANSFORMERS_CACHE="./cache/transformers"
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
print_status $? "Environment variables set"

# Create a run script
echo "Creating run script..."
cat > run_cdrc.sh << 'EOF'
#!/bin/bash
# Activate environment
source cdrc_env/bin/activate

# Set environment variables
export HF_HOME="./cache/huggingface"
export TRANSFORMERS_CACHE="./cache/transformers"
export CUDA_VISIBLE_DEVICES=0  # Adjust as needed

# Run the experiment
python experiment_runner.py --config experiment_config.json --step all
EOF

chmod +x run_cdrc.sh
print_status $? "Run script created"

echo ""
echo "============================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Download Jigsaw dataset if not already done"
echo "2. Adjust experiment_config.json as needed"
echo "3. Run the experiment:"
echo "   ./run_cdrc.sh"
echo ""
echo "Or run individual steps:"
echo "   source cdrc_env/bin/activate"
echo "   python experiment_runner.py --step preprocess"
echo ""