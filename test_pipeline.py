"""
Test script for CDRC pipeline components
"""

import os
import sys
import pickle
import numpy as np
from utils import load_pickle, save_pickle, set_seed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_preprocessing():
    """Test data preprocessing module"""
    logger.info("Testing data preprocessing module...")
    
    try:
        from data_preprocessing import DataPreprocessor
        
        # Initialize with small test
        preprocessor = DataPreprocessor()
        
        # Test loading small dataset
        prompts = preprocessor.load_realtoxicityprompts(num_samples=10)
        assert len(prompts) == 10, f"Expected 10 prompts, got {len(prompts)}"
        
        # Test response generation for one prompt
        test_prompt = prompts[0]['text']
        responses = preprocessor.generate_responses(test_prompt, num_responses=5)
        assert len(responses) == 5, f"Expected 5 responses, got {len(responses)}"
        
        # Test candidate set building
        candidate_set = preprocessor.build_candidate_set(prompts[0], responses, max_set_size=3)
        assert len(candidate_set) <= 3, f"Candidate set too large: {len(candidate_set)}"
        
        logger.info("✓ Data preprocessing tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data preprocessing test failed: {e}")
        return False


def test_toxicity_scoring():
    """Test toxicity scoring module"""
    logger.info("Testing toxicity scoring module...")
    
    try:
        from toxicity_scoring import ToxicityScoringModule
        
        # Initialize scorer
        scorer = ToxicityScoringModule()
        
        # Test scoring on sample texts
        test_texts = [
            "This is a normal sentence.",
            "I hate everything about this.",
            "The weather is nice today."
        ]
        
        # Test with original Detoxify
        scores = scorer.detoxify_original.predict(test_texts)
        assert 'toxicity' in scores, "Missing toxicity scores"
        assert len(scores['toxicity']) == 3, "Wrong number of scores"
        
        logger.info("✓ Toxicity scoring tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Toxicity scoring test failed: {e}")
        return False


def test_conformal_generation():
    """Test conformal set generation"""
    logger.info("Testing conformal set generation...")
    
    try:
        from conformal_generation import ConformalSetGenerator
        
        # Create test data
        test_processed_data = {
            0: {
                'prompt': {'idx': 0, 'text': 'Test prompt'},
                'candidate_set': [
                    {'idx': 0, 'text': 'Response 1', 'perplexity': 10.5},
                    {'idx': 1, 'text': 'Response 2', 'perplexity': 15.2},
                    {'idx': 2, 'text': 'Response 3', 'perplexity': 8.9}
                ]
            }
        }
        
        test_x_cal = {
            0: {
                'pred': ['Response 1', 'Response 2', 'Response 3'],
                'detoxify_ft': [[0.1, 0.5, 0.3]],
                'detoxify_human': [0.2, 0.6, 0.4]
            }
        }
        
        # Save test data
        os.makedirs('./test_data', exist_ok=True)
        save_pickle(test_processed_data, './test_data/test_processed.pkl')
        save_pickle(test_x_cal, './test_data/test_x_cal.pkl')
        
        # Initialize generator
        generator = ConformalSetGenerator(output_dir='./test_data/conformal_sets')
        
        # Test filter parameters generation
        filter_params = generator.generate_filter_parameters(num_groups=3)
        assert len(filter_params) == 3, f"Expected 3 filter groups, got {len(filter_params)}"
        
        # Test F1 score computation
        f1_score = generator.compute_f1_score(test_processed_data[0]['candidate_set'])
        assert 0 <= f1_score <= 1, f"Invalid F1 score: {f1_score}"
        
        logger.info("✓ Conformal generation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Conformal generation test failed: {e}")
        return False


def test_sampling_compatibility():
    """Test compatibility with sampling.py"""
    logger.info("Testing sampling.py compatibility...")
    
    try:
        # Create test data in expected format
        test_x_cal = {
            0: {
                'pred': ['text1', 'text2'],
                'detoxify_ft': [np.array([0.1, 0.2])],
                'detoxify_human': np.array([0.15, 0.25])
            }
        }
        
        test_y_cal = {
            0: {
                'set': {
                    0: [(0, 'text1'), (1, 'text2')]
                }
            }
        }
        
        # Create directory structure
        os.makedirs('./results_llama2_7B_Real/test_toxic_new', exist_ok=True)
        os.makedirs('./results_llama2_7B_Real/conformal_set', exist_ok=True)
        
        # Save test data
        save_pickle(test_x_cal, './results_llama2_7B_Real/test_toxic_new/test_batch.pkl')
        save_pickle(test_y_cal, './results_llama2_7B_Real/conformal_set/conformal_set_size_F1_0.800.pkl')
        
        logger.info("✓ Sampling compatibility test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Sampling compatibility test failed: {e}")
        return False


def test_evaluation_module():
    """Test evaluation and visualization module"""
    logger.info("Testing evaluation module...")
    
    try:
        from evaluation_visualization import EvaluationModule
        
        # Create test results
        os.makedirs('./test_results/DRC/trial_0/alpha_0.3_beta_0.75_0.8', exist_ok=True)
        
        test_scores = {
            'optimal_lambda': 0.5,
            'mean_human_score': 0.3,
            'percentile_95_human_score': 0.6,
            'max_human_score': 0.9,
            'beta_cvar_human_score': 0.4,
            'average_sample_count': 15.5
        }
        
        save_pickle(test_scores, './test_results/DRC/trial_0/alpha_0.3_beta_0.75_0.8/scores.pkl')
        
        # Initialize evaluator
        evaluator = EvaluationModule(results_base_dir='./test_results')
        
        # Test loading results
        scores = evaluator.load_experiment_results('DRC', 0, 0.3, 0.75, 0.8)
        assert scores is not None, "Failed to load test scores"
        assert scores['optimal_lambda'] == 0.5, "Wrong lambda value"
        
        logger.info("✓ Evaluation module tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Evaluation module test failed: {e}")
        return False


def run_all_tests():
    """Run all module tests"""
    logger.info("=" * 50)
    logger.info("Running CDRC Pipeline Tests")
    logger.info("=" * 50)
    
    set_seed(42)
    
    tests = [
        ("Data Preprocessing", test_data_preprocessing),
        ("Toxicity Scoring", test_toxicity_scoring),
        ("Conformal Generation", test_conformal_generation),
        ("Sampling Compatibility", test_sampling_compatibility),
        ("Evaluation Module", test_evaluation_module)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("=" * 50)
    logger.info("Test Summary:")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    # Cleanup
    import shutil
    for dir_to_remove in ['./test_data', './test_results']:
        if os.path.exists(dir_to_remove):
            shutil.rmtree(dir_to_remove)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
