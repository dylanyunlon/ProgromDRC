#!/bin/bash
# Batch experiment runner for CDRC framework

# Activate environment
source cdrc_env/bin/activate

# Set experiment parameters
ALPHA_VALUES=(0.2 0.25 0.3 0.35 0.4)
BETA_VALUES=(0.6 0.7 0.75 0.8 0.85)
F1_SCORES=(0.750 0.800 0.850)

# Function to run single experiment
run_single_experiment() {
    local method=$1
    local trial=$2
    local alpha=$3
    local beta=$4
    local f1=$5
    
    echo "Running: Method=$method Trial=$trial Alpha=$alpha Beta=$beta F1=$f1"
    
    if [ "$method" == "DRC" ]; then
        python sampling.py --trial_index $trial --f1_score $f1 --alpha $alpha --beta $beta
    elif [ "$method" == "DKW" ]; then
        python sampling.py --trial_index $trial --f1_score $f1 --alpha $alpha --beta $beta --use_dkw True
    elif [ "$method" == "BJ" ]; then
        python sampling_var.py --trial_index $trial --f1_score $f1 --alpha $alpha --beta $beta --use_bj True
    fi
}

# Main execution
echo "Starting CDRC experiments..."
echo "=============================="

# Run preprocessing if needed
if [ ! -f "./processed_data/processed_data_complete.pkl" ]; then
    echo "Running data preprocessing..."
    python experiment_runner.py --step preprocess
fi

# Run toxicity scoring if needed
if [ ! -f "./processed_data/x_cal_scored.pkl" ]; then
    echo "Running toxicity scoring..."
    python experiment_runner.py --step score
fi

# Run conformal generation if needed
if [ ! -d "./cdrc_experiments/conformal_sets" ] || [ -z "$(ls -A ./cdrc_experiments/conformal_sets)" ]; then
    echo "Generating conformal sets..."
    python experiment_runner.py --step conformal
fi

# Prepare test data
echo "Preparing test data..."
python -c "
from experiment_runner import ExperimentRunner
import pickle
runner = ExperimentRunner()
with open('./processed_data/x_cal_scored.pkl', 'rb') as f:
    x_cal_scored = pickle.load(f)
runner.prepare_test_data(x_cal_scored, {})
"

# Run experiments for each method
for method in "DRC" "DKW" "BJ"; do
    echo "Running $method experiments..."
    
    # Set number of trials based on method
    if [ "$method" == "BJ" ]; then
        NUM_TRIALS=3
    else
        NUM_TRIALS=15
    fi
    
    # Run trials
    for trial in $(seq 0 $((NUM_TRIALS-1))); do
        for alpha in "${ALPHA_VALUES[@]}"; do
            for beta in "${BETA_VALUES[@]}"; do
                for f1 in "${F1_SCORES[@]}"; do
                    run_single_experiment $method $trial $alpha $beta $f1
                done
            done
        done
    done
done

# Run evaluation
echo "Running evaluation..."
python experiment_runner.py --step evaluate

echo "=============================="
echo "All experiments completed!"
