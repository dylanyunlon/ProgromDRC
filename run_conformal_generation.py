#!/usr/bin/env python
"""
Generate conformal sets for science domain
"""

import os
import pickle
import logging
from conformal_generation import ScienceConformalSetGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate conformal sets"""
    
    # Paths
    processed_data_path = "./processed_science_data/processed_science_data_complete.pkl"
    x_cal_path = "./processed_science_data/x_cal_scored.pkl"
    
    # Check files exist
    if not os.path.exists(processed_data_path):
        logger.error(f"Processed data not found: {processed_data_path}")
        return
    
    if not os.path.exists(x_cal_path):
        logger.error(f"Scored data not found: {x_cal_path}")
        logger.info("Please run run_science_scoring.py first!")
        return
    
    # Initialize generator
    logger.info("Initializing conformal set generator...")
    generator = ScienceConformalSetGenerator(
        output_dir="./results_science/conformal_set"
    )
    
    # Generate conformal sets
    logger.info("Generating conformal sets...")
    all_conformal_sets = generator.generate_conformal_sets(
        processed_data_path,
        x_cal_path
    )
    
    # Save conformal sets
    logger.info("Saving conformal sets...")
    saved_files = generator.save_conformal_sets_by_f1(all_conformal_sets)
    
    # Analyze results
    logger.info("Analyzing conformal sets...")
    analysis = generator.analyze_conformal_sets(all_conformal_sets)
    
    logger.info(f"\nGenerated {len(all_conformal_sets)} conformal set configurations")
    logger.info(f"Saved files: {saved_files}")
    
    logger.info("\n✓ Conformal set generation complete!")


if __name__ == "__main__":
    main()
