#!/usr/bin/env python
"""
CDRC Experiment Runner for Scientific Text Generation
Runs conformal risk control experiments with different methods (DRC, DKW, BJ)
Based on original sampling.py and sampling_var.py implementations
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import time
import random
import uuid
from scipy.stats import scoreatpercentile, spearmanr
from scipy.interpolate import interp1d

# Import CDRC modules
from berk_jones import berk_jones
from utils import (
    save_pickle, load_pickle, save_json, load_json,
    set_seed, format_time, calculate_statistics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScienceCDRCExperiment:
    """Main experiment runner for science domain CDRC"""
    
    def __init__(self, config_path: str):
        """Initialize experiment with configuration"""
        self.config = load_json(config_path)
        
        # Set paths
        self.results_dir = self.config['paths']['results_dir']
        self.var_results_dir = self.config['paths']['var_results_dir']
        self.data_dir = self.config['paths']['data_dir']
        self.conformal_sets_dir = os.path.join(self.results_dir, '..', 'conformal_set')
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.var_results_dir, exist_ok=True)
        
        logger.info(f"Initialized experiment: {self.config['experiment']['name']}")
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    def load_data(self) -> Tuple[Dict, Dict]:
        """Load calibration data and conformal sets"""
        # Load scored calibration data
        cal_file = os.path.join(self.data_dir, 'calibration_data_scored.pkl')
        if not os.path.exists(cal_file):
            # Try without _scored suffix
            cal_file = os.path.join(self.data_dir, 'calibration_data.pkl')
            if not os.path.exists(cal_file):
                raise FileNotFoundError(f"Calibration data not found")
        
        with open(cal_file, 'rb') as f:
            data = pickle.load(f)
            x_cal = data['x_cal']
            y_cal = data['y_cal']
        
        # Convert to expected format for science domain
        # Original format expects 'detoxify_ft' and 'detoxify_human', we have 'science_validity_scores'
        for key in x_cal:
            if 'science_validity_scores' in x_cal[key]:
                # Convert to expected format
                x_cal[key]['detoxify_ft'] = [np.array(x_cal[key]['science_validity_scores'])]
                x_cal[key]['detoxify_human'] = x_cal[key].get('science_validity_scores', 
                                                               x_cal[key]['science_validity_scores'])
        
        logger.info(f"Loaded calibration data: {len(x_cal)} prompts")
        
        return x_cal, y_cal
    
    def distortion_risk_control(self, x_cal, y_cal, alpha, beta):
        """DRC method implementation (from sampling.py)"""
        lambda_candidates = np.linspace(0.0210, 0.8560, 1000)
        risks = []
        
        for lambda_ in tqdm(lambda_candidates, desc="DRC optimization"):
            r_lambdas = []
            for key, val in x_cal.items():
                scores = val['detoxify_ft'][0].reshape(-1)
                human_scores = val['detoxify_human']
                C_all = y_cal[0]['set'][key] if 0 in y_cal and 'set' in y_cal[0] else []
                
                # For science domain, LOWER scores are better (higher validity)
                # So we flip the comparison
                C_lambda_ = [(idx, val) for idx, val in C_all if scores[idx] >= lambda_]
                
                if len(C_lambda_) == 0:
                    r_lambdas.append(1.0)  # Worst score for science
                else:
                    adjusted_scores = [human_scores[idx] for idx, _ in C_lambda_]
                    r_lambda_ = min(adjusted_scores)  # Best (lowest) score
                    r_lambdas.append(r_lambda_)
            
            var_r_lambda = np.percentile(r_lambdas, beta * 100)
            empirical_risk = np.mean([r for r in r_lambdas if r < var_r_lambda])
            max_values = np.minimum(r_lambdas, var_r_lambda)
            
            sigma_lambda = 1/(1-beta)*np.std(max_values)
            risks.append(empirical_risk - 1.645*sigma_lambda/np.sqrt(len(r_lambdas)))
        
        risks = np.array(risks)
        valid_lambdas = lambda_candidates[risks >= alpha]
        if valid_lambdas.size > 0:
            lambda_optimal = np.min(valid_lambdas)  # Changed to min for science
        else:
            lambda_optimal = None
        
        return lambda_optimal
    
    def distortion_risk_control_DKW(self, x_cal, y_cal, alpha, beta, n_samples):
        """DKW method implementation"""
        epsilon = np.sqrt(np.log(1 / 0.05) / (2 * n_samples))
        
        lambda_candidates = np.linspace(0.0210, 0.8560, 1000)
        risks = []
        
        for lambda_ in tqdm(lambda_candidates, desc="DKW optimization"):
            r_lambdas = []
            for key, val in x_cal.items():
                scores = val['detoxify_ft'][0].reshape(-1)
                human_scores = val['detoxify_human']
                C_all = y_cal[0]['set'][key] if 0 in y_cal and 'set' in y_cal[0] else []
                
                # Science domain: higher scores are better
                C_lambda_ = [(idx, val) for idx, val in C_all if scores[idx] >= lambda_]
                
                if len(C_lambda_) == 0:
                    r_lambdas.append(1.0)
                else:
                    adjusted_scores = [human_scores[idx] for idx, _ in C_lambda_]
                    r_lambda_ = min(adjusted_scores)
                    r_lambdas.append(r_lambda_)
            
            n = len(r_lambdas)
            n_beta = min(int(np.ceil(n*(beta+epsilon)))-1, n-1)
            sorted_scores = np.sort(r_lambdas)
            empirical_cvar = (n_beta/n-beta-epsilon)*sorted_scores[n_beta] + \
                           1/n*np.sum([sorted_scores[i] for i in range(n_beta+1,n)]) + \
                           epsilon*sorted_scores[-1]
            risks.append(empirical_cvar/(1-beta))
        
        valid_lambdas = lambda_candidates[np.array(risks) >= alpha]
        if valid_lambdas.size > 0:
            lambda_optimal = np.min(valid_lambdas)
        else:
            lambda_optimal = None
        
        return lambda_optimal
    
    def distortion_risk_control_BJ(self, x_cal, y_cal, alpha, beta, n_samples):
        """Berk-Jones method implementation"""
        lambda_candidates = np.linspace(0.0210, 0.8560, 1000)
        risks = []
        
        for lambda_ in tqdm(lambda_candidates, desc="BJ optimization"):
            r_lambdas = []
            for key, val in x_cal.items():
                scores = val['detoxify_ft'][0].reshape(-1)
                human_scores = val['detoxify_human']
                C_all = y_cal[0]['set'][key] if 0 in y_cal and 'set' in y_cal[0] else []
                
                C_lambda_ = [(idx, val) for idx, val in C_all if scores[idx] >= lambda_]
                
                if len(C_lambda_) == 0:
                    r_lambdas.append(1.0)
                else:
                    adjusted_scores = [human_scores[idx] for idx, _ in C_lambda_]
                    r_lambda_ = min(adjusted_scores)
                    r_lambdas.append(r_lambda_)
            
            n = len(r_lambdas)
            LB = berk_jones(n, 0.05)
            
            n_beta = n
            for i, item in enumerate(LB):
                if item >= beta:
                    n_beta = i
                    break
            
            sorted_scores = np.sort(r_lambdas)
            
            result = 0
            for i in range(n_beta, n):
                if i == 0:
                    diff = LB[i] - beta
                else:
                    diff = LB[i] - LB[i - 1]
                result += diff * sorted_scores[i]
            
            result += (1 - LB[-1]) * 1
            risks.append(result/(1-beta))
        
        valid_lambdas = lambda_candidates[np.array(risks) >= alpha]
        if valid_lambdas.size > 0:
            lambda_optimal = np.min(valid_lambdas)
        else:
            lambda_optimal = None
        
        return lambda_optimal
    
    def evaluate_remaining_data(self, remaining_x_cal, lambda_optimal, remaining_y_cal, beta):
        """Evaluate on test data"""
        if lambda_optimal is None:
            return None, None, None, None, None
        
        science_scores_all = []
        sample_counts = []
        science_scores_max = []
        
        for key, val in remaining_x_cal.items():
            scores = val['detoxify_ft'][0].reshape(-1)
            human_scores = val['detoxify_human']
            C_all = remaining_y_cal[0]['set'][key] if 0 in remaining_y_cal and 'set' in remaining_y_cal[0] else []
            
            sample_count = 0
            for idx, val in C_all:
                sample_count += 1
                if scores[idx] >= lambda_optimal:
                    sample_counts.append(sample_count)
                    break
            
            C_lambda_selected = [(idx, val, scores[idx], human_scores[idx]) 
                               for idx, val in C_all if scores[idx] >= lambda_optimal]
            
            science_scores_all.extend([item[3] for item in C_lambda_selected])
            if C_lambda_selected:
                science_scores_max.append(min([item[3] for item in C_lambda_selected]))
            else:
                science_scores_max.append(1.0)
        
        sorted_scores = np.sort(science_scores_max)
        threshold_index = int(np.ceil(beta * len(sorted_scores))) - 1
        beta_cvar_score = np.mean(sorted_scores[:threshold_index+1])
        
        mean_score = np.mean(science_scores_all)
        percentile_95_score = scoreatpercentile(science_scores_all, 5)  # 5th percentile for science
        best_score = np.min(science_scores_all)
        average_sample_count = np.mean(sample_counts) if sample_counts else 0
        
        return mean_score, percentile_95_score, best_score, beta_cvar_score, average_sample_count
    
    def run_single_experiment(self, method: str, trial: int, alpha: float, beta: float, 
                            f1_score: float, bias_level: float = 0.15) -> Dict:
        """Run a single experiment"""
        logger.info(f"Running {method} trial {trial}: α={alpha}, β={beta}, bias={bias_level}")
        
        # Load data
        x_cal, y_cal = self.load_data()
        
        # Handle bias levels
        if bias_level != 0.15:
            # Use biased scores if available
            bias_field = f'science_validity_biased_{int(bias_level*100)}'
            for key in x_cal:
                if bias_field in x_cal[key]:
                    x_cal[key]['detoxify_ft'] = [np.array(x_cal[key][bias_field])]
                    x_cal[key]['detoxify_human'] = x_cal[key][bias_field]
        
        # Load appropriate conformal set
        conformal_file = os.path.join(
            self.conformal_sets_dir,
            f"conformal_set_size_F1_{f1_score:.3f}.pkl"
        )
        
        if not os.path.exists(conformal_file):
            # Try all_conformal_sets.pkl
            all_sets_file = os.path.join(self.conformal_sets_dir, "all_conformal_sets.pkl")
            if os.path.exists(all_sets_file):
                with open(all_sets_file, 'rb') as f:
                    all_sets = pickle.load(f)
                # Use first available group
                y_cal_conf = {0: all_sets[list(all_sets.keys())[0]]}
            else:
                logger.warning(f"Conformal set not found for F1={f1_score}")
                y_cal_conf = y_cal
        else:
            with open(conformal_file, 'rb') as f:
                y_cal_conf = pickle.load(f)
        
        # Split data
        all_keys = list(x_cal.keys())
        random.seed(int(uuid.uuid4().int % (2**32)) + trial)
        random.shuffle(all_keys)
        
        split_idx = int(len(all_keys) * self.config['data']['train_test_split'])
        train_keys = all_keys[:split_idx]
        test_keys = all_keys[split_idx:]
        
        x_cal_train = {k: x_cal[k] for k in train_keys}
        y_cal_train = {0: {'set': {k: y_cal_conf[0]['set'][k] for k in train_keys 
                                  if k in y_cal_conf[0]['set']}}}
        
        x_cal_test = {k: x_cal[k] for k in test_keys}
        y_cal_test = {0: {'set': {k: y_cal_conf[0]['set'][k] for k in test_keys 
                                 if k in y_cal_conf[0]['set']}}}
        
        # Run method
        start_time = time.time()
        
        if method == "DRC":
            lambda_optimal = self.distortion_risk_control(x_cal_train, y_cal_train, alpha, beta)
        elif method == "DKW":
            lambda_optimal = self.distortion_risk_control_DKW(x_cal_train, y_cal_train, 
                                                             alpha, beta, len(train_keys))
        elif method == "BJ":
            lambda_optimal = self.distortion_risk_control_BJ(x_cal_train, y_cal_train, 
                                                           alpha, beta, len(train_keys))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Evaluate on test set
        mean_score, percentile_95, best_score, beta_cvar, avg_samples = \
            self.evaluate_remaining_data(x_cal_test, lambda_optimal, y_cal_test, beta)
        
        runtime = time.time() - start_time
        
        # Save results
        if method == "BJ":
            base_folder = f"{self.var_results_dir}/{method}/trial_{trial}/alpha_{alpha}_beta_{beta}_{f1_score:.3f}"
        else:
            base_folder = f"{self.results_dir}/{method}/trial_{trial}/alpha_{alpha}_beta_{beta}_{f1_score:.3f}"
        
        os.makedirs(base_folder, exist_ok=True)
        
        results = {
            'method': method,
            'trial': trial,
            'alpha': alpha,
            'beta': beta,
            'f1_score': f1_score,
            'bias_level': bias_level,
            'lambda_optimal': lambda_optimal,
            'mean_score': mean_score,
            'percentile_95_score': percentile_95,
            'best_score': best_score,
            'beta_cvar_score': beta_cvar,
            'average_sample_count': avg_samples,
            'runtime': runtime
        }
        
        # Save results
        save_pickle(results, os.path.join(base_folder, 'scores.pkl'))
        
        # Save summary
        with open(os.path.join(base_folder, 'results_summary.txt'), 'w') as f:
            f.write(f"Lambda Optimal: {lambda_optimal}\n")
            f.write(f"Mean Science Score: {mean_score}\n")
            f.write(f"5th Percentile Science Score: {percentile_95}\n")
            f.write(f"Best Science Score: {best_score}\n")
            f.write(f"Beta CVaR Science Score: {beta_cvar}\n")
            f.write(f"Average Sample Count: {avg_samples}\n")
            f.write(f"Runtime: {format_time(runtime)}\n")
        
        return results
    
    def run_all_experiments(self):
        """Run all experiments based on configuration"""
        all_results = defaultdict(list)
        
        # Experiment parameters
        methods = self.config['experiments']['methods']
        alpha_values = self.config['parameters']['alpha_values']
        beta_values = self.config['parameters']['beta_values']
        f1_thresholds = self.config['parameters']['f1_thresholds']
        bias_levels = self.config['models']['bias_levels']
        
        # Run experiments
        for method in methods:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {method} experiments")
            logger.info(f"{'='*50}")
            
            num_trials = self.config['experiments']['num_trials'][method]
            
            for trial in range(num_trials):
                for alpha in alpha_values:
                    for beta in beta_values:
                        for f1 in f1_thresholds:
                            for bias in bias_levels:
                                try:
                                    result = self.run_single_experiment(
                                        method, trial, alpha, beta, f1, bias
                                    )
                                    all_results[method].append(result)
                                    
                                    # Save intermediate results
                                    self.save_results(all_results)
                                    
                                except Exception as e:
                                    logger.error(f"Failed {method} trial {trial}: {e}")
                                    import traceback
                                    traceback.print_exc()
        
        # Final save and report
        self.save_results(all_results)
        self.generate_summary_report(all_results)
        
        logger.info("\n✓ All experiments completed!")
        return all_results
    
    def save_results(self, results: Dict):
        """Save experiment results"""
        results_file = os.path.join(self.results_dir, 'all_experiment_results.pkl')
        save_pickle(results, results_file)
        
        # Save summary as JSON
        summary = {}
        for method, method_results in results.items():
            summary[method] = []
            for result in method_results:
                if result is not None:
                    summary_item = {
                        'method': result['method'],
                        'alpha': result['alpha'],
                        'beta': result['beta'],
                        'bias_level': result['bias_level'],
                        'mean_score': result['mean_score'],
                        'beta_cvar_score': result['beta_cvar_score'],
                        'runtime': result['runtime']
                    }
                    summary[method].append(summary_item)
        
        summary_file = os.path.join(self.results_dir, 'experiment_summary.json')
        save_json(summary, summary_file)
    
    def generate_summary_report(self, results: Dict):
        """Generate summary report"""
        report_lines = [
            "CDRC Science Experiments Summary Report",
            "=" * 50,
            f"Experiment: {self.config['experiment']['name']}",
            f"Date: {pd.Timestamp.now()}",
            "",
            "Results Summary:",
            "-" * 30
        ]
        
        for method, method_results in results.items():
            report_lines.append(f"\n{method} Method:")
            
            # Group by bias level
            bias_groups = defaultdict(list)
            for result in method_results:
                if result is not None:
                    bias_groups[result['bias_level']].append(result)
            
            for bias_level, bias_results in bias_groups.items():
                report_lines.append(f"\n  Bias Level: {bias_level}")
                
                if bias_results:
                    # Find best configuration (highest science score = lowest value)
                    best_result = min(bias_results, 
                                    key=lambda r: r['mean_score'] if r['mean_score'] is not None else float('inf'))
                    
                    report_lines.append(f"  Best configuration:")
                    report_lines.append(f"    Alpha: {best_result['alpha']}")
                    report_lines.append(f"    Beta: {best_result['beta']}")
                    if best_result['mean_score'] is not None:
                        report_lines.append(f"    Avg Science Score: {best_result['mean_score']:.3f}")
        
        # Save report
        report_file = os.path.join(self.results_dir, 'experiment_report.txt')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY:")
        print("="*50)
        for line in report_lines[-20:]:
            print(line)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run CDRC experiments for science domain")
    parser.add_argument('--config', type=str, default='experiment_config.json',
                       help='Path to experiment configuration file')
    parser.add_argument('--method', type=str, default=None,
                       help='Run only specific method (DRC/BJ/DKW)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with reduced parameters')
    parser.add_argument('--trial', type=int, default=None,
                       help='Run only specific trial')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Run only specific alpha value')
    parser.add_argument('--beta', type=float, default=None,
                       help='Run only specific beta value')
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    # Modify config for test mode
    if args.test:
        config = load_json(args.config)
        config['parameters']['alpha_values'] = [0.3]
        config['parameters']['beta_values'] = [0.7]
        config['models']['bias_levels'] = [0.15]
        config['experiments']['num_trials'] = {'DRC': 1, 'DKW': 1, 'BJ': 1}
        
        # Save test config
        test_config_path = 'test_experiment_config.json'
        save_json(config, test_config_path)
        args.config = test_config_path
        logger.info("Running in TEST mode with reduced parameters")
    
    # Run experiments
    experiment = ScienceCDRCExperiment(args.config)
    
    # Handle specific method/trial/parameter requests
    if args.method or args.trial is not None or args.alpha or args.beta:
        config = load_json(args.config)
        
        if args.method:
            config['experiments']['methods'] = [args.method]
        
        if args.trial is not None:
            for method in config['experiments']['methods']:
                config['experiments']['num_trials'][method] = 1
        
        if args.alpha:
            config['parameters']['alpha_values'] = [args.alpha]
        
        if args.beta:
            config['parameters']['beta_values'] = [args.beta]
        
        # Update experiment config
        experiment.config = config
        
        # Run single experiment if all parameters specified
        if args.method and args.trial is not None and args.alpha and args.beta:
            # Run single experiment
            f1 = config['parameters']['f1_thresholds'][0]
            bias = config['models']['bias_levels'][0]
            
            result = experiment.run_single_experiment(
                args.method, args.trial, args.alpha, args.beta, f1, bias
            )
            
            logger.info(f"\nSingle experiment completed:")
            logger.info(f"Mean score: {result['mean_score']}")
            logger.info(f"Runtime: {format_time(result['runtime'])}")
            return
    
    # Run all experiments
    results = experiment.run_all_experiments()
    
    logger.info("\n✓ Experiment runner completed successfully!")


if __name__ == "__main__":
    main()