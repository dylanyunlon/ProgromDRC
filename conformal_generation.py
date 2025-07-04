"""
Conformal Set Generation Module for CDRC Framework - Science Version
Handles generation of conformal sets with scientific quality filtering
"""

import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScienceConformalSetGenerator:
    def __init__(self, output_dir="./results_science/conformal_set"):
        """
        Initialize conformal set generator for science domain
        
        Args:
            output_dir: Directory to save conformal sets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_filter_parameters(self, num_groups=5):
        """
        Generate different filter parameter groups for scientific quality
        
        Args:
            num_groups: Number of parameter groups to generate
            
        Returns:
            List of filter parameter dictionaries
        """
        filter_params = []
        
        # Group 0: Baseline (minimal filtering)
        filter_params.append({
            'group_id': 0,
            'min_validity_score': 0.0,
            'min_coherence': 0.0,
            'require_formula': False,
            'require_units': False,
            'max_similarity': 1.0,
            'min_diversity': 0.0,
            'description': 'Baseline - minimal filtering'
        })
        
        # Generate other groups with increasing scientific rigor
        for i in range(1, num_groups):
            filter_params.append({
                'group_id': i,
                'min_validity_score': i * 0.15,  # 0.15, 0.30, 0.45, 0.60
                'min_coherence': i * 0.1,        # 0.1, 0.2, 0.3, 0.4
                'require_formula': i >= 3,        # Required for high-rigor groups
                'require_units': i >= 2,          # Required for medium+ rigor
                'max_similarity': 1.0 - i * 0.1,  # 0.9, 0.8, 0.7, 0.6
                'min_diversity': i * 0.05,        # 0.05, 0.10, 0.15, 0.20
                'description': f'Filter group {i} - {"low" if i < 2 else "medium" if i < 4 else "high"} scientific rigor'
            })
        
        return filter_params
    
    def apply_scientific_filters(self, candidate_set, filter_params, validity_scores=None):
        """
        Apply scientific quality filtering to a candidate set
        
        Args:
            candidate_set: List of candidates with scores
            filter_params: Dictionary of filter parameters
            validity_scores: Optional pre-computed validity scores
            
        Returns:
            Filtered candidate set
        """
        if filter_params['group_id'] == 0:
            # Minimal filtering for baseline
            return candidate_set
        
        filtered_set = []
        
        for i, candidate in enumerate(candidate_set):
            # Check validity score if available
            if validity_scores and filter_params['min_validity_score'] > 0:
                if validity_scores[i] < filter_params['min_validity_score']:
                    continue
            
            # Check coherence
            if candidate.get('coherence', 1.0) < filter_params['min_coherence']:
                continue
            
            # Check formula requirement
            if filter_params['require_formula'] and not candidate.get('has_formula', False):
                continue
            
            # Check units requirement
            if filter_params['require_units'] and not candidate.get('has_units', False):
                continue
            
            filtered_set.append(candidate)
        
        # Apply diversity filtering
        if filter_params['min_diversity'] > 0 and len(filtered_set) > 1:
            diverse_set = self._select_diverse_subset(
                filtered_set, 
                filter_params['max_similarity'],
                filter_params['min_diversity']
            )
            filtered_set = diverse_set
        
        return filtered_set
    
    def _select_diverse_subset(self, candidates, max_similarity, min_diversity):
        """
        Select diverse subset based on text similarity
        
        Args:
            candidates: List of candidates
            max_similarity: Maximum allowed similarity
            min_diversity: Minimum required diversity
            
        Returns:
            Diverse subset of candidates
        """
        if len(candidates) <= 1:
            return candidates
        
        selected = [candidates[0]]  # Start with first candidate
        
        for candidate in candidates[1:]:
            is_diverse = True
            
            for selected_cand in selected:
                # Simple character-level similarity for efficiency
                similarity = self._text_similarity(
                    candidate.get('text', ''), 
                    selected_cand.get('text', '')
                )
                
                if similarity > max_similarity:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(candidate)
        
        # Check if we achieved minimum diversity
        avg_diversity = self._compute_set_diversity(selected)
        if avg_diversity < min_diversity and len(selected) > 2:
            # Remove most similar pairs until diversity threshold met
            while avg_diversity < min_diversity and len(selected) > 2:
                # Find most similar pair
                max_sim = 0
                remove_idx = -1
                
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        sim = self._text_similarity(
                            selected[i].get('text', ''),
                            selected[j].get('text', '')
                        )
                        if sim > max_sim:
                            max_sim = sim
                            remove_idx = j
                
                if remove_idx >= 0:
                    selected.pop(remove_idx)
                else:
                    break
                
                avg_diversity = self._compute_set_diversity(selected)
        
        return selected
    
    def _text_similarity(self, text1, text2):
        """
        Calculate text similarity (Jaccard similarity on word level)
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _compute_set_diversity(self, candidate_set):
        """
        Compute average diversity (1 - similarity) for a set
        """
        if len(candidate_set) <= 1:
            return 1.0
        
        similarities = []
        for i in range(len(candidate_set)):
            for j in range(i + 1, len(candidate_set)):
                sim = self._text_similarity(
                    candidate_set[i].get('text', ''), 
                    candidate_set[j].get('text', '')
                )
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity
    
    def compute_scientific_f1_score(self, candidate_set, field='general', 
                                  validity_scores=None, reference_set=None):
        """
        Compute F1 score for scientific quality of candidate set
        
        Args:
            candidate_set: Generated candidate set
            field: Scientific field
            validity_scores: Science validity scores if available
            reference_set: Reference set for comparison (if available)
            
        Returns:
            F1 score
        """
        if not candidate_set:
            return 0.0
        
        # Components of F1 score for scientific text
        scores = {}
        
        # 1. Scientific validity (if scores available)
        if validity_scores:
            scores['validity'] = np.mean(validity_scores)
        else:
            # Heuristic based on features
            formula_ratio = sum(1 for c in candidate_set if c.get('has_formula', False)) / len(candidate_set)
            unit_ratio = sum(1 for c in candidate_set if c.get('has_units', False)) / len(candidate_set)
            scores['validity'] = (formula_ratio + unit_ratio) / 2
        
        # 2. Coherence
        coherences = [c.get('coherence', 0.5) for c in candidate_set]
        scores['coherence'] = np.mean(coherences)
        
        # 3. Diversity
        scores['diversity'] = self._compute_set_diversity(candidate_set)
        
        # 4. Coverage (size relative to maximum)
        scores['coverage'] = min(len(candidate_set) / 32, 1.0)
        
        # 5. Field appropriateness
        if field == 'chemistry':
            chem_keywords = ['reaction', 'compound', 'molecule', 'element']
            field_score = sum(
                1 for c in candidate_set 
                if any(kw in c.get('text', '').lower() for kw in chem_keywords)
            ) / len(candidate_set)
        elif field == 'physics':
            phys_keywords = ['force', 'energy', 'momentum', 'particle']
            field_score = sum(
                1 for c in candidate_set 
                if any(kw in c.get('text', '').lower() for kw in phys_keywords)
            ) / len(candidate_set)
        else:
            field_score = 0.5
        scores['field_appropriateness'] = field_score
        
        # Weighted combination
        weights = {
            'validity': 0.35,
            'coherence': 0.25,
            'diversity': 0.20,
            'coverage': 0.10,
            'field_appropriateness': 0.10
        }
        
        f1_score = sum(scores[k] * weights[k] for k in weights)
        
        return f1_score
    
    def generate_conformal_sets(self, processed_data_path, x_cal_path=None):
        """
        Generate conformal sets for all filter parameter groups
        
        Args:
            processed_data_path: Path to processed data with candidate sets
            x_cal_path: Path to x_cal data with scores (optional)
            
        Returns:
            Dictionary of conformal sets for each parameter group
        """
        # Load data
        with open(processed_data_path, 'rb') as f:
            processed_data = pickle.load(f)
        
        # Load scores if available
        validity_scores_dict = {}
        if x_cal_path and os.path.exists(x_cal_path):
            with open(x_cal_path, 'rb') as f:
                x_cal = pickle.load(f)
            
            # Extract validity scores
            for idx, data in x_cal.items():
                if 'science_validity_scores' in data:
                    validity_scores_dict[idx] = data['science_validity_scores']
        
        # Generate filter parameters
        filter_params_list = self.generate_filter_parameters()
        
        # Store results for each parameter group
        all_conformal_sets = {}
        
        for filter_params in tqdm(filter_params_list, desc="Generating scientific conformal sets"):
            group_id = filter_params['group_id']
            logger.info(f"Processing filter group {group_id}: {filter_params['description']}")
            
            conformal_sets = {}
            f1_scores = []
            set_sizes = []
            
            for idx in processed_data.keys():
                # Get candidate set
                candidate_set = processed_data[idx]['candidate_set']
                field = processed_data[idx].get('field', 'general')
                
                # Get validity scores if available
                validity_scores = validity_scores_dict.get(idx, None)
                
                # Apply scientific filters
                filtered_set = self.apply_scientific_filters(
                    candidate_set, 
                    filter_params,
                    validity_scores
                )
                
                # Format for compatibility with sampling.py
                formatted_set = [(i, cand['text']) for i, cand in enumerate(filtered_set)]
                conformal_sets[idx] = formatted_set
                
                # Compute F1 score
                f1_score = self.compute_scientific_f1_score(
                    filtered_set, 
                    field,
                    validity_scores[:len(filtered_set)] if validity_scores else None
                )
                f1_scores.append(f1_score)
                set_sizes.append(len(filtered_set))
            
            # Calculate average metrics
            avg_f1 = np.mean(f1_scores)
            avg_size = np.mean(set_sizes)
            
            # Store results
            all_conformal_sets[group_id] = {
                'filter_params': filter_params,
                'set': conformal_sets,
                'avg_f1_score': avg_f1,
                'avg_set_size': avg_size,
                'f1_scores': f1_scores,
                'set_sizes': set_sizes
            }
            
            logger.info(f"Group {group_id} - Avg F1: {avg_f1:.3f}, Avg Size: {avg_size:.1f}")
        
        return all_conformal_sets
    
    def save_conformal_sets_by_f1(self, all_conformal_sets):
        """
        Save conformal sets organized by F1 score
        
        Args:
            all_conformal_sets: Dictionary of conformal sets for each parameter group
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        # Create mapping of F1 scores to filenames
        f1_to_filename = {
            0.750: "conformal_set_size_F1_0.750.pkl",
            0.800: "conformal_set_size_F1_0.800.pkl", 
            0.850: "conformal_set_size_F1_0.850.pkl"
        }
        
        for group_id, data in all_conformal_sets.items():
            avg_f1 = data['avg_f1_score']
            
            # Find closest F1 target
            closest_f1 = min(f1_to_filename.keys(), key=lambda x: abs(x - avg_f1))
            
            # Use the corresponding filename
            filename = f1_to_filename[closest_f1]
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the conformal set
            with open(filepath, 'wb') as f:
                pickle.dump({group_id: data}, f)
            
            saved_files.append(filepath)
            logger.info(f"Saved conformal set (F1={avg_f1:.3f}) to {filepath}")
        
        # Also save all sets in one file
        all_sets_file = os.path.join(self.output_dir, "all_conformal_sets.pkl")
        with open(all_sets_file, 'wb') as f:
            pickle.dump(all_conformal_sets, f)
        
        logger.info(f"Saved all conformal sets to {all_sets_file}")
        
        return saved_files
    
    def analyze_conformal_sets(self, all_conformal_sets):
        """
        Analyze properties of generated conformal sets
        
        Args:
            all_conformal_sets: Dictionary of conformal sets
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        for group_id, data in all_conformal_sets.items():
            conformal_sets = data['set']
            
            # Calculate detailed statistics
            set_sizes = data['set_sizes']
            f1_scores = data['f1_scores']
            
            group_analysis = {
                'filter_params': data['filter_params'],
                'avg_f1_score': float(data['avg_f1_score']),
                'num_prompts': len(conformal_sets),
                'avg_set_size': float(np.mean(set_sizes)),
                'std_set_size': float(np.std(set_sizes)),
                'min_set_size': int(np.min(set_sizes)),
                'max_set_size': int(np.max(set_sizes)),
                'median_set_size': float(np.median(set_sizes)),
                'empty_sets': int(sum(1 for s in set_sizes if s == 0)),
                'f1_distribution': {
                    'mean': float(np.mean(f1_scores)),
                    'std': float(np.std(f1_scores)),
                    'min': float(np.min(f1_scores)),
                    'max': float(np.max(f1_scores)),
                    'percentiles': {
                        '25': float(np.percentile(f1_scores, 25)),
                        '50': float(np.percentile(f1_scores, 50)),
                        '75': float(np.percentile(f1_scores, 75))
                    }
                }
            }
            
            analysis[group_id] = group_analysis
            
            # Print summary
            logger.info(f"\nGroup {group_id} Analysis:")
            logger.info(f"  Filter: {data['filter_params']['description']}")
            logger.info(f"  Avg F1 Score: {group_analysis['avg_f1_score']:.3f}")
            logger.info(f"  Avg Set Size: {group_analysis['avg_set_size']:.1f} ± {group_analysis['std_set_size']:.1f}")
            logger.info(f"  Size Range: [{group_analysis['min_set_size']}, {group_analysis['max_set_size']}]")
            logger.info(f"  Empty Sets: {group_analysis['empty_sets']}")
        
        # Save analysis
        analysis_file = os.path.join(self.output_dir, "conformal_set_analysis.json")
        with open(analysis_file, 'w') as f:
            # Convert any remaining numpy types to Python native types
            def convert_to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(v) for v in obj]
                else:
                    return obj
            
            json.dump(convert_to_native(analysis), f, indent=2)
        
        return analysis


# Backward compatibility wrapper
class ConformalSetGenerator(ScienceConformalSetGenerator):
    """Wrapper for backward compatibility"""
    pass


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = ScienceConformalSetGenerator()
    
    # Generate conformal sets
    all_conformal_sets = generator.generate_conformal_sets(
        processed_data_path="./processed_science_data/processed_science_data_complete.pkl",
        x_cal_path="./processed_science_data/x_cal_scored.pkl"
    )
    
    # Save conformal sets
    saved_files = generator.save_conformal_sets_by_f1(all_conformal_sets)
    
    # Analyze results
    analysis = generator.analyze_conformal_sets(all_conformal_sets)
    
    print(f"\nGenerated {len(all_conformal_sets)} conformal set configurations")
    print(f"Saved to: {saved_files}")