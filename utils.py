"""
Utility functions for CDRC framework
"""

import os
import pickle
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def save_pickle(data: Any, filepath: str) -> None:
    """Save data to pickle file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved data to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load data from pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def save_json(data: Dict, filepath: str) -> None:
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved data to {filepath}")


def load_json(filepath: str) -> Dict:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def format_time(seconds: float) -> str:
    """Format time in seconds to readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values"""
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values)
    }


def merge_results_files(file_patterns: List[str], output_file: str) -> None:
    """Merge multiple result files into one"""
    all_results = {}
    
    for pattern in file_patterns:
        if os.path.exists(pattern):
            data = load_pickle(pattern)
            all_results.update(data)
    
    save_pickle(all_results, output_file)
    logger.info(f"Merged {len(file_patterns)} files into {output_file}")


def validate_experiment_config(config: Dict) -> bool:
    """Validate experiment configuration"""
    required_keys = [
        'experiment', 'data', 'models', 'parameters', 
        'experiments', 'paths', 'compute'
    ]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
    
    # Validate paths exist or can be created
    for path_key, path_value in config['paths'].items():
        if path_key.endswith('_dir'):
            try:
                os.makedirs(path_value, exist_ok=True)
            except Exception as e:
                logger.error(f"Cannot create directory {path_value}: {e}")
                return False
    
    return True


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
    }


def batch_iterator(data: List[Any], batch_size: int):
    """Create batches from a list"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Specific utility functions for CDRC

def format_results_for_latex(results_df) -> str:
    """Format results DataFrame for LaTeX table"""
    latex_str = results_df.to_latex(
        index=False,
        float_format="%.3f",
        column_format='l' + 'c' * (len(results_df.columns) - 1)
    )
    return latex_str


def check_data_compatibility(x_cal: Dict, y_cal: Dict) -> bool:
    """Check if x_cal and y_cal data are compatible"""
    x_keys = set(x_cal.keys())
    
    # Check y_cal structure
    if not y_cal or 0 not in y_cal:
        logger.error("y_cal missing key 0")
        return False
    
    y_keys = set(y_cal[0]['set'].keys()) if 'set' in y_cal[0] else set()
    
    # Check if keys match
    if x_keys != y_keys:
        logger.error(f"Key mismatch: x_cal has {len(x_keys)} keys, y_cal has {len(y_keys)} keys")
        missing_in_y = x_keys - y_keys
        missing_in_x = y_keys - x_keys
        if missing_in_y:
            logger.error(f"Keys in x_cal but not in y_cal: {list(missing_in_y)[:5]}")
        if missing_in_x:
            logger.error(f"Keys in y_cal but not in x_cal: {list(missing_in_x)[:5]}")
        return False
    
    # Check data format for a sample
    sample_key = list(x_keys)[0]
    if 'pred' not in x_cal[sample_key]:
        logger.error("x_cal entries missing 'pred' field")
        return False
    
    return True


def create_experiment_summary(config: Dict, results: Dict) -> Dict:
    """Create a summary of experiment configuration and results"""
    summary = {
        'config': {
            'dataset': config['data']['dataset'],
            'num_prompts': config['data']['num_prompts'],
            'methods': config['experiments']['methods'],
            'alpha_range': [min(config['parameters']['alpha_values']), 
                           max(config['parameters']['alpha_values'])],
            'beta_range': [min(config['parameters']['beta_values']), 
                          max(config['parameters']['beta_values'])],
        },
        'results': results,
        'timestamp': np.datetime64('now').astype(str)
    }
    return summary
