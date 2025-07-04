#!/usr/bin/env python
"""
Run science feasibility scoring on processed data
"""

import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
from science_feasibility_scoring import ScienceFeasibilityScorer, BiasedScienceScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Score all candidate sets"""
    
    # Load calibration data
    cal_file = './processed_science_data/calibration_data.pkl'
    with open(cal_file, 'rb') as f:
        data = pickle.load(f)
        x_cal = data['x_cal']
        y_cal = data['y_cal']
    
    logger.info(f"Loaded {len(x_cal)} prompts for scoring")
    
    # Initialize scorer
    logger.info("Initializing science feasibility scorer...")
    scorer = ScienceFeasibilityScorer(
        base_model="allenai/scibert_scivocab_uncased",
        use_gpt4=False
    )
    
    # Score all candidate sets
    logger.info("Scoring candidate sets...")
    
    for idx in tqdm(x_cal.keys(), desc="Scoring prompts"):
        prompt = x_cal[idx]['prompt']
        field = x_cal[idx]['field']
        candidates = x_cal[idx]['pred']
        
        # Score each candidate
        scores = []
        detailed_results = []
        
        for candidate in candidates:
            try:
                result = scorer.evaluate_scientific_validity(
                    prompt,
                    candidate,
                    field
                )
                scores.append(result.overall_score)
                detailed_results.append(result)
            except Exception as e:
                logger.warning(f"Scoring failed for candidate: {e}")
                scores.append(0.5)  # Default score
                detailed_results.append(None)
        
        # Store scores
        x_cal[idx]['science_validity_scores'] = scores
        x_cal[idx]['science_detailed_results'] = detailed_results
    
    # Create biased scorers for robustness testing
    logger.info("Creating biased scorers...")
    bias_levels = [0.15, 0.30, 0.70]
    
    for bias_level in bias_levels:
        logger.info(f"Scoring with bias level {bias_level}")
        biased_scorer = scorer.create_biased_evaluator(bias_level)
        
        # Score with biased model
        for idx in tqdm(x_cal.keys(), desc=f"Biased scoring ({bias_level})"):
            prompt = x_cal[idx]['prompt']
            field = x_cal[idx]['field']
            candidates = x_cal[idx]['pred']
            
            biased_scores = []
            for candidate in candidates:
                try:
                    score = biased_scorer.score(prompt, candidate, field)
                    biased_scores.append(score)
                except:
                    biased_scores.append(0.5)
            
            x_cal[idx][f'science_validity_biased_{int(bias_level*100)}'] = biased_scores
    
    # Save scored data
    output_file = './processed_science_data/x_cal_scored.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(x_cal, f)
    
    logger.info(f"Saved scored data to {output_file}")
    
    # Save complete calibration data with scores
    scored_cal_file = './processed_science_data/calibration_data_scored.pkl'
    with open(scored_cal_file, 'wb') as f:
        pickle.dump({'x_cal': x_cal, 'y_cal': y_cal}, f)
    
    # Print statistics
    logger.info("\nScoring Statistics:")
    all_scores = []
    for idx in x_cal:
        all_scores.extend(x_cal[idx]['science_validity_scores'])
    
    logger.info(f"Total responses scored: {len(all_scores)}")
    logger.info(f"Average score: {np.mean(all_scores):.3f}")
    logger.info(f"Score std: {np.std(all_scores):.3f}")
    logger.info(f"Min score: {np.min(all_scores):.3f}")
    logger.info(f"Max score: {np.max(all_scores):.3f}")
    
    # Violations summary
    all_violations = []
    for idx in x_cal:
        if 'science_detailed_results' in x_cal[idx]:
            for result in x_cal[idx]['science_detailed_results']:
                if result and result.violations:
                    all_violations.extend(result.violations)
    
    if all_violations:
        logger.info(f"\nTotal violations found: {len(all_violations)}")
        violation_types = {}
        for v in all_violations:
            v_type = v.split(':')[0]
            violation_types[v_type] = violation_types.get(v_type, 0) + 1
        logger.info("Violation types:")
        for v_type, count in violation_types.items():
            logger.info(f"  {v_type}: {count}")
    
    logger.info("\n✓ Scoring complete!")


if __name__ == "__main__":
    main()
