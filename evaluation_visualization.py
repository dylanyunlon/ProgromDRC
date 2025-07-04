"""
Evaluation and Visualization Module for CDRC Framework
Handles performance evaluation and result visualization
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationModule:
    def __init__(self, results_base_dir="./results_detoxify_0.15"):
        """
        Initialize evaluation module
        
        Args:
            results_base_dir: Base directory for results
        """
        self.results_base_dir = results_base_dir
        self.methods = ['DRC', 'DKW', 'BJ']
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def load_experiment_results(self, method, trial, alpha, beta, f1_score):
        """
        Load results from a single experiment
        
        Args:
            method: Method name (DRC, DKW, BJ)
            trial: Trial index
            alpha: Alpha parameter
            beta: Beta parameter
            f1_score: F1 score identifier
            
        Returns:
            Dictionary of results
        """
        folder = f"{self.results_base_dir}/{method}/trial_{trial}/alpha_{alpha}_beta_{beta}_{f1_score}"
        scores_file = os.path.join(folder, "scores.pkl")
        
        if not os.path.exists(scores_file):
            logger.warning(f"Results file not found: {scores_file}")
            return None
        
        with open(scores_file, 'rb') as f:
            scores = pickle.load(f)
        
        return scores
    
    def aggregate_results(self, alpha_values, beta_values, f1_scores, num_trials):
        """
        Aggregate results across all experiments
        
        Args:
            alpha_values: List of alpha values tested
            beta_values: List of beta values tested
            f1_scores: List of F1 scores tested
            num_trials: Dictionary mapping method to number of trials
            
        Returns:
            Aggregated results DataFrame
        """
        all_results = []
        
        for method in self.methods:
            n_trials = num_trials.get(method, 15)
            
            for alpha in alpha_values:
                for beta in beta_values:
                    for f1 in f1_scores:
                        method_results = {
                            'method': method,
                            'alpha': alpha,
                            'beta': beta,
                            'f1_score': f1,
                            'cvar_scores': [],
                            'var_scores': [],
                            'sample_counts': [],
                            'lambda_values': []
                        }
                        
                        for trial in range(n_trials):
                            scores = self.load_experiment_results(
                                method, trial, alpha, beta, f1
                            )
                            
                            if scores:
                                if 'beta_cvar_human_score' in scores:
                                    method_results['cvar_scores'].append(scores['beta_cvar_human_score'])
                                elif 'var_human_score' in scores:
                                    method_results['var_scores'].append(scores['var_human_score'])
                                
                                if scores['average_sample_count']:
                                    method_results['sample_counts'].append(scores['average_sample_count'])
                                if scores['optimal_lambda']:
                                    method_results['lambda_values'].append(scores['optimal_lambda'])
                        
                        # Calculate statistics
                        if method_results['cvar_scores']:
                            method_results['mean_cvar'] = np.mean(method_results['cvar_scores'])
                            method_results['std_cvar'] = np.std(method_results['cvar_scores'])
                            method_results['coverage_cvar'] = np.mean(
                                [s <= alpha for s in method_results['cvar_scores']]
                            )
                        
                        if method_results['var_scores']:
                            method_results['mean_var'] = np.mean(method_results['var_scores'])
                            method_results['std_var'] = np.std(method_results['var_scores'])
                            method_results['coverage_var'] = np.mean(
                                [s <= alpha for s in method_results['var_scores']]
                            )
                        
                        if method_results['sample_counts']:
                            method_results['mean_samples'] = np.mean(method_results['sample_counts'])
                            method_results['std_samples'] = np.std(method_results['sample_counts'])
                        
                        all_results.append(method_results)
        
        return pd.DataFrame(all_results)
    
    def plot_coverage_vs_alpha(self, results_df, save_path="./figures/coverage_vs_alpha.png"):
        """
        Plot coverage probability vs target risk level (Figure 4)
        
        Args:
            results_df: Aggregated results DataFrame
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # CVaR plot
        for method in self.methods:
            method_data = results_df[results_df['method'] == method]
            if 'mean_cvar' in method_data.columns:
                alpha_values = sorted(method_data['alpha'].unique())
                coverage_values = []
                
                for alpha in alpha_values:
                    alpha_data = method_data[method_data['alpha'] == alpha]
                    if not alpha_data.empty and 'coverage_cvar' in alpha_data.columns:
                        coverage_values.append(alpha_data['coverage_cvar'].mean())
                    else:
                        coverage_values.append(0)
                
                ax1.plot(alpha_values, coverage_values, marker='o', label=f'CDRC-{method[0]}')
        
        # Add diagonal reference line
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Target')
        ax1.set_xlabel('Target Risk Level (α)')
        ax1.set_ylabel('Empirical Coverage')
        ax1.set_title('CVaR Coverage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # VaR plot
        for method in self.methods:
            method_data = results_df[results_df['method'] == method]
            if 'mean_var' in method_data.columns:
                alpha_values = sorted(method_data['alpha'].unique())
                coverage_values = []
                
                for alpha in alpha_values:
                    alpha_data = method_data[method_data['alpha'] == alpha]
                    if not alpha_data.empty and 'coverage_var' in alpha_data.columns:
                        coverage_values.append(alpha_data['coverage_var'].mean())
                    else:
                        coverage_values.append(0)
                
                ax2.plot(alpha_values, coverage_values, marker='o', label=f'CDRC-{method[0]}')
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Target')
        ax2.set_xlabel('Target Risk Level (α)')
        ax2.set_ylabel('Empirical Coverage')
        ax2.set_title('VaR Coverage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved coverage plot to {save_path}")
    
    def plot_risk_vs_samples(self, results_df, save_path="./figures/risk_vs_samples.png"):
        """
        Plot realized risk vs average sample count (Figure 5)
        
        Args:
            results_df: Aggregated results DataFrame
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for method in self.methods:
            method_data = results_df[results_df['method'] == method]
            
            if 'mean_samples' in method_data.columns and 'mean_cvar' in method_data.columns:
                # Group by sample count bins
                sample_counts = method_data['mean_samples'].values
                risks = method_data['mean_cvar'].values
                
                # Remove NaN values
                mask = ~(np.isnan(sample_counts) | np.isnan(risks))
                sample_counts = sample_counts[mask]
                risks = risks[mask]
                
                if len(sample_counts) > 0:
                    ax.scatter(sample_counts, risks, label=f'CDRC-{method[0]}', alpha=0.7, s=50)
        
        ax.set_xlabel('Average Sample Count')
        ax.set_ylabel('Realized CVaR')
        ax.set_title('Risk-Efficiency Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved risk vs samples plot to {save_path}")
    
    def plot_beta_sensitivity(self, results_df, save_path="./figures/beta_sensitivity.png"):
        """
        Plot sensitivity to beta parameter (Figure 6)
        
        Args:
            results_df: Aggregated results DataFrame
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Fix alpha and vary beta
        fixed_alpha = 0.35
        alpha_data = results_df[results_df['alpha'] == fixed_alpha]
        
        for method in self.methods:
            method_data = alpha_data[alpha_data['method'] == method]
            
            if not method_data.empty:
                beta_values = sorted(method_data['beta'].unique())
                mean_risks = []
                std_risks = []
                
                for beta in beta_values:
                    beta_data = method_data[method_data['beta'] == beta]
                    if 'mean_cvar' in beta_data.columns:
                        mean_risks.append(beta_data['mean_cvar'].mean())
                        std_risks.append(beta_data['std_cvar'].mean())
                    else:
                        mean_risks.append(np.nan)
                        std_risks.append(np.nan)
                
                # Plot with error bars
                ax.errorbar(beta_values, mean_risks, yerr=std_risks, 
                           marker='o', capsize=5, label=f'CDRC-{method[0]}')
        
        ax.axhline(y=fixed_alpha, color='r', linestyle='--', alpha=0.5, label=f'Target α={fixed_alpha}')
        ax.set_xlabel('Beta (β)')
        ax.set_ylabel('Realized CVaR')
        ax.set_title(f'Sensitivity to β Parameter (α={fixed_alpha})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved beta sensitivity plot to {save_path}")
    
    def generate_performance_table(self, results_df, save_path="./tables/performance_summary.csv"):
        """
        Generate performance summary table
        
        Args:
            results_df: Aggregated results DataFrame
            save_path: Path to save table
        """
        summary_data = []
        
        for method in self.methods:
            method_data = results_df[results_df['method'] == method]
            
            if not method_data.empty:
                summary = {
                    'Method': f'CDRC-{method[0]}',
                    'Avg Coverage (CVaR)': f"{method_data['coverage_cvar'].mean():.3f}" if 'coverage_cvar' in method_data else 'N/A',
                    'Avg Coverage (VaR)': f"{method_data['coverage_var'].mean():.3f}" if 'coverage_var' in method_data else 'N/A',
                    'Avg Sample Count': f"{method_data['mean_samples'].mean():.1f}" if 'mean_samples' in method_data else 'N/A',
                    'Avg Risk (CVaR)': f"{method_data['mean_cvar'].mean():.3f}" if 'mean_cvar' in method_data else 'N/A',
                    'Avg Risk (VaR)': f"{method_data['mean_var'].mean():.3f}" if 'mean_var' in method_data else 'N/A'
                }
                summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        summary_df.to_csv(save_path, index=False)
        
        logger.info(f"Saved performance summary to {save_path}")
        print("\nPerformance Summary:")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def analyze_human_machine_alignment(self, ft_scores_path, human_scores_path):
        """
        Analyze human-machine alignment with different bias levels
        
        Args:
            ft_scores_path: Path to machine scores
            human_scores_path: Path to human scores
            
        Returns:
            Alignment statistics
        """
        with open(ft_scores_path, 'rb') as f:
            ft_scores = pickle.load(f)
        
        with open(human_scores_path, 'rb') as f:
            human_scores = pickle.load(f)
        
        from scipy.stats import spearmanr
        
        # Calculate correlation
        corr, p_value = spearmanr(ft_scores, human_scores)
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(ft_scores, human_scores, alpha=0.5, s=10)
        plt.xlabel('Machine Toxicity Score')
        plt.ylabel('Human Toxicity Score')
        plt.title(f'Human-Machine Alignment\nSpearman ρ = {corr:.3f} (p = {p_value:.2e})')
        
        # Add diagonal reference
        lims = [0, 1]
        plt.plot(lims, lims, 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('./figures/human_machine_alignment.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'correlation': corr, 'p_value': p_value}


# Example usage
if __name__ == "__main__":
    # Initialize evaluation module
    evaluator = EvaluationModule()
    
    # Define experiment parameters
    alpha_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    beta_values = [0.6, 0.7, 0.75, 0.8, 0.85]
    f1_scores = [0.750, 0.800, 0.850]  # Example F1 scores
    num_trials = {'DRC': 15, 'DKW': 15, 'BJ': 3}
    
    # Aggregate results
    results_df = evaluator.aggregate_results(alpha_values, beta_values, f1_scores, num_trials)
    
    # Generate visualizations
    evaluator.plot_coverage_vs_alpha(results_df)
    evaluator.plot_risk_vs_samples(results_df)
    evaluator.plot_beta_sensitivity(results_df)
    
    # Generate performance table
    evaluator.generate_performance_table(results_df)
    
    # Analyze human-machine alignment
    if os.path.exists('all_ft_scores.pkl') and os.path.exists('all_human_scores.pkl'):
        alignment_stats = evaluator.analyze_human_machine_alignment(
            'all_ft_scores.pkl', 
            'all_human_scores.pkl'
        )
        print(f"\nHuman-Machine Alignment: ρ = {alignment_stats['correlation']:.3f}")