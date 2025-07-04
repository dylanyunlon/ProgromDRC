# CDRC Framework - Requirements and Setup Guide

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM for LLaMA-2-7B)
- 32GB+ RAM
- 100GB+ free disk space

## Dependencies

### Core Dependencies
```bash
# Create virtual environment
python -m venv cdrc_env
source cdrc_env/bin/activate  # On Windows: cdrc_env\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install core packages
pip install numpy==1.24.3
pip install scipy==1.10.1
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install tqdm==4.65.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
```

### NLP and Model Dependencies
```bash
# Transformers and related
pip install transformers==4.30.2
pip install datasets==2.13.1
pip install tokenizers==0.13.3
pip install accelerate==0.20.3

# Toxicity detection
pip install detoxify==0.5.1

# Text similarity
pip install rouge-score==0.1.2

# Additional utilities
pip install crossprob  # For Berk-Jones implementation
```

### Optional Dependencies
```bash
# For parallel processing
pip install joblib==1.3.1

# For experiment tracking
pip install wandb  # optional
pip install tensorboard  # optional
```

## Data Setup

### 1. REALTOXICITYPROMPTS Dataset
The dataset will be automatically downloaded via HuggingFace datasets library.

### 2. Jigsaw Toxic Comment Dataset
You need to manually download the Jigsaw dataset from Kaggle:

1. Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Download `train.csv.zip`
3. Extract and place `train.csv` in `./jigsaw_data/`

```bash
mkdir -p jigsaw_data
# Place train.csv in this directory
```

### 3. Human Toxicity Annotations (Optional)
If you have human toxicity annotations, structure them as:
```python
{
    prompt_idx: {
        'toxicity': [scores_for_each_response]
    }
}
```
Save as `human_toxicity_scores.pkl`

## Directory Structure

Create the following directory structure:
```
cdrc_project/
├── sampling.py              # Your provided files
├── sampling_var.py
├── berk_jones.py
├── data_preprocessing.py    # New modules
├── toxicity_scoring.py
├── conformal_generation.py
├── evaluation_visualization.py
├── experiment_runner.py
├── experiment_config.json
├── requirements.txt
├── jigsaw_data/
│   └── train.csv
├── cache/                   # Model cache
├── processed_data/          # Processed datasets
├── biased_models/           # Trained toxicity models
├── results_llama2_7B_Real/  # Expected by sampling.py
│   ├── test_toxic_new/
│   └── conformal_set/
├── results_detoxify_0.15/   # Results directory
├── var_results_detoxify_0.15/
├── figures/                 # Generated plots
└── tables/                  # Result tables
```

## Environment Configuration

### 1. GPU Memory Management
For LLaMA-2-7B with limited GPU memory:
```python
# Add to your scripts
import torch
torch.cuda.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    # Your inference code
```

### 2. HuggingFace Cache
Set cache directory:
```bash
export HF_HOME=/path/to/large/disk/huggingface_cache
export TRANSFORMERS_CACHE=/path/to/large/disk/transformers_cache
```

### 3. Batch Processing
Adjust batch sizes based on your GPU:
- 16GB GPU: batch_size=8-16
- 24GB GPU: batch_size=16-32
- 40GB+ GPU: batch_size=32-64

## Running the Pipeline

### Full Pipeline
```bash
python experiment_runner.py --config experiment_config.json --step all
```

### Individual Steps
```bash
# Step 1: Data preprocessing
python experiment_runner.py --step preprocess

# Step 2: Toxicity scoring
python experiment_runner.py --step score

# Step 3: Conformal set generation
python experiment_runner.py --step conformal

# Step 4: Sampling experiments
python experiment_runner.py --step sample

# Step 5: Evaluation
python experiment_runner.py --step evaluate
```

### Running Specific Experiments
```bash
# Run single CDRC-L experiment
python sampling.py --trial_index 0 --f1_score 0.8 --alpha 0.35 --beta 0.75

# Run DKW variant
python sampling.py --trial_index 0 --f1_score 0.8 --alpha 0.35 --beta 0.75 --use_dkw True

# Run Berk-Jones variant
python sampling_var.py --trial_index 0 --f1_score 0.8 --alpha 0.35 --beta 0.75 --use_bj True
```

## Experiment Configuration

Edit `experiment_config.json` to customize:
```json
{
  "data": {
    "num_prompts": 10000,      // Number of prompts to process
    "num_responses_per_prompt": 40,  // Responses per prompt
    "train_test_split": 0.6    // 60% for calibration, 40% for test
  },
  "parameters": {
    "alpha_values": [0.2, 0.25, 0.3, 0.35, 0.4],  // Risk levels
    "beta_values": [0.6, 0.7, 0.75, 0.8, 0.85],   // CVaR/VaR levels
    "f1_thresholds": [0.750, 0.800, 0.850]        // Quality thresholds
  },
  "experiments": {
    "num_trials": {
      "DRC": 15,   // 15 trials for CDRC-L
      "DKW": 15,   // 15 trials for CDRC-DKW
      "BJ": 3      // 3 trials for CDRC-BJ (computationally expensive)
    }
  }
}
```

## Troubleshooting

### Out of Memory Errors
1. Reduce batch size in config
2. Use gradient checkpointing
3. Process data in smaller chunks
4. Use CPU offloading for large models

### Slow Processing
1. Enable parallel processing in config
2. Use GPU acceleration
3. Pre-compute and cache intermediate results
4. Reduce number of responses per prompt for testing

### Missing Dependencies
```bash
# Check all dependencies
pip list

# Install missing packages
pip install -r requirements.txt
```

## Monitoring Progress

The pipeline provides detailed logging:
- Progress bars for long operations
- Step-by-step status updates
- Error messages with context
- Summary statistics at each stage

Results are saved incrementally, allowing resumption if interrupted.

## Expected Outputs

After successful completion:
1. **Processed Data**: `./processed_data/processed_data_complete.pkl`
2. **Scored Data**: `./processed_data/x_cal_scored.pkl`
3. **Conformal Sets**: `./cdrc_experiments/conformal_sets/conformal_set_size_F1_*.pkl`
4. **Results**: `./results_detoxify_0.15/[method]/trial_*/alpha_*_beta_*_*/scores.pkl`
5. **Figures**: `./figures/coverage_vs_alpha.png`, `risk_vs_samples.png`, etc.
6. **Tables**: `./tables/performance_summary.csv`

## Citation

If you use this implementation, please cite the original CDRC paper and acknowledge the implementation.