#!/usr/bin/env python
"""
Test science feasibility scoring
"""

import os
import pickle
import logging
from science_feasibility_scoring import ScienceFeasibilityScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_scoring():
    """Test the scoring module"""
    
    # Load calibration data
    cal_file = './processed_science_data/calibration_data.pkl'
    if not os.path.exists(cal_file):
        logger.error(f"Calibration file not found: {cal_file}")
        return
    
    with open(cal_file, 'rb') as f:
        data = pickle.load(f)
        x_cal = data['x_cal']
        y_cal = data['y_cal']
    
    logger.info(f"Loaded {len(x_cal)} prompts for scoring")
    
    # Initialize scorer
    logger.info("Initializing science feasibility scorer...")
    scorer = ScienceFeasibilityScorer(
        base_model="allenai/scibert_scivocab_uncased",
        use_gpt4=False  # Set to True if you have GPT-4 API key
    )
    
    # Test on first prompt
    first_idx = list(x_cal.keys())[0]
    test_data = x_cal[first_idx]
    
    logger.info(f"\nTesting on prompt: {test_data['prompt']}")
    logger.info(f"Field: {test_data['field']}")
    logger.info(f"Number of candidates: {len(test_data['pred'])}")
    
    # Score first candidate
    if test_data['pred']:
        first_response = test_data['pred'][0]
        logger.info(f"\nFirst response: {first_response[:100]}...")
        
        result = scorer.evaluate_scientific_validity(
            test_data['prompt'],
            first_response,
            test_data['field']
        )
        
        logger.info(f"\nScoring results:")
        logger.info(f"  Overall score: {result.overall_score:.3f}")
        logger.info(f"  Logical coherence: {result.logical_coherence:.3f}")
        logger.info(f"  Terminology accuracy: {result.terminology_accuracy:.3f}")
        logger.info(f"  Formula correctness: {result.formula_correctness:.3f}")
        logger.info(f"  Violations: {result.violations}")
        logger.info(f"  Confidence: {result.confidence:.3f}")
    
    logger.info("\n✓ Scoring test complete!")


if __name__ == "__main__":
    test_scoring()
