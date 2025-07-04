"""
Data Preprocessing Module for CDRC Framework - Science Version
Handles loading science prompts, generating continuations, and building candidate sets
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from collections import defaultdict
import logging
import re
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScienceDataPreprocessor:
    def __init__(self, 
                 model_name="facebook/galactica-1.3b",  # Science-focused model
                 device="cuda", 
                 cache_dir="./cache",
                 use_multiple_models=False):
        """
        Initialize the science data preprocessor
        
        Args:
            model_name: HuggingFace model identifier (Galactica recommended for science)
            device: Device to run the model on
            cache_dir: Directory to cache datasets and models
            use_multiple_models: Whether to use multiple models for diversity
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.use_multiple_models = use_multiple_models
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize primary model
        logger.info(f"Loading primary model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Set padding token - Galactica specific handling
        if self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                # For Galactica, use the specific padding token
                self.tokenizer.pad_token = "<pad>"
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<pad>")
                if self.tokenizer.pad_token_id == self.tokenizer.unk_token_id:
                    # If <pad> doesn't exist, add it
                    self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        # Initialize additional models if requested
        if use_multiple_models:
            self._load_additional_models()
        
        # Initialize ROUGE scorer for similarity
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Science-specific patterns
        self.formula_pattern = re.compile(r'(\$[^\$]+\$|\\\[[^\]]+\\\]|[A-Z][a-z]?\d*(?:\s*[+-]\s*[A-Z][a-z]?\d*)*)')
        self.unit_pattern = re.compile(r'\d+\s*(K|°C|°F|m/s|N|J|mol|M|mL|L|g|kg|Pa|atm)')
        
    def _load_additional_models(self):
        """Load additional models for generation diversity"""
        self.additional_models = []
        
        model_names = [
            "EleutherAI/gpt-neo-1.3B",  # General model
            "allenai/scibert_scivocab_uncased",  # Science understanding
        ]
        
        for model_name in model_names:
            try:
                logger.info(f"Loading additional model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                if tokenizer.pad_token is None:
                    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                    else:
                        tokenizer.add_special_tokens({'pad_token': '<pad>'})
                    
                self.additional_models.append((model, tokenizer))
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")
    
    def load_science_prompts(self, data_file: str, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load science prompts from preprocessed file
        
        Args:
            data_file: Path to science prompts file (from download_science_data.py)
            num_samples: Number of samples to load (None for all)
            
        Returns:
            List of prompts with metadata
        """
        logger.info(f"Loading science prompts from {data_file}...")
        
        with open(data_file, 'rb') as f:
            prompts = pickle.load(f)
        
        if num_samples:
            prompts = prompts[:num_samples]
        
        logger.info(f"Loaded {len(prompts)} prompts")
        
        # Print field distribution
        fields = [p.get('field', 'general') for p in prompts]
        field_counts = {field: fields.count(field) for field in set(fields)}
        logger.info(f"Field distribution: {field_counts}")
        
        return prompts
    
    def generate_responses(self, 
                         prompt_text: str, 
                         field: str = "general",
                         num_responses: int = 40, 
                         max_new_tokens: int = 100, 
                         temperature: float = 0.8, 
                         top_p: float = 0.9,
                         use_field_guidance: bool = True) -> List[str]:
        """
        Generate multiple scientific continuations for a prompt
        
        Args:
            prompt_text: Input prompt
            field: Scientific field (chemistry/physics/general)
            num_responses: Number of responses to generate
            max_new_tokens: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_field_guidance: Whether to add field-specific guidance
            
        Returns:
            List of generated responses
        """
        # Add field-specific guidance if requested
        if use_field_guidance and field != "general":
            if field == "chemistry":
                guided_prompt = f"[Chemistry] {prompt_text}"
            elif field == "physics":
                guided_prompt = f"[Physics] {prompt_text}"
            else:
                guided_prompt = prompt_text
        else:
            guided_prompt = prompt_text
        
        # Tokenize input
        inputs = self.tokenizer(guided_prompt, return_tensors="pt", padding=True).to(self.device)
        
        # Remove token_type_ids if present (for models like Galactica/OPT that don't use them)
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        input_length = inputs['input_ids'].shape[1]
        
        responses = []
        
        # Generate using primary model
        primary_responses = self._generate_with_model(
            self.model, 
            self.tokenizer, 
            inputs, 
            num_responses if not self.use_multiple_models else num_responses // 2,
            max_new_tokens,
            temperature,
            top_p
        )
        responses.extend(primary_responses)
        
        # Generate using additional models if available
        if self.use_multiple_models and self.additional_models:
            remaining = num_responses - len(responses)
            responses_per_model = remaining // len(self.additional_models)
            
            for model, tokenizer in self.additional_models:
                model_inputs = tokenizer(guided_prompt, return_tensors="pt", padding=True).to(self.device)
                model_responses = self._generate_with_model(
                    model,
                    tokenizer,
                    model_inputs,
                    responses_per_model,
                    max_new_tokens,
                    temperature,
                    top_p
                )
                responses.extend(model_responses)
        
        # Remove the prompt and field guidance from responses
        cleaned_responses = []
        for response in responses:
            # Remove field guidance if added
            if use_field_guidance and field != "general":
                response = response.replace(f"[{field.capitalize()}] ", "")
            
            # Remove original prompt
            if response.startswith(prompt_text):
                response = response[len(prompt_text):].strip()
            
            cleaned_responses.append(response)
        
        return cleaned_responses[:num_responses]  # Ensure exact number requested
    
    def _generate_with_model(self, model, tokenizer, inputs, num_responses, 
                           max_new_tokens, temperature, top_p) -> List[str]:
        """Generate responses with a specific model"""
        responses = []
        
        # Remove token_type_ids if not used by the model (e.g., Galactica)
        if 'token_type_ids' in inputs and not hasattr(model.config, 'type_vocab_size'):
            inputs.pop('token_type_ids')
        
        with torch.no_grad():
            for _ in range(num_responses):
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                responses.append(generated_text)
        
        return responses
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of a text using the language model
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        # Remove token_type_ids if present (for models like Galactica/OPT)
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        with torch.no_grad():
            # Use inputs['input_ids'] instead of inputs.input_ids
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            
        return perplexity
    
    def calculate_scientific_coherence(self, prompt: str, continuation: str, field: str) -> float:
        """
        Calculate scientific coherence score
        
        Args:
            prompt: Original prompt
            continuation: Generated continuation
            field: Scientific field
            
        Returns:
            Coherence score (0-1)
        """
        # Basic ROUGE similarity
        rouge_score = self.calculate_rouge_similarity(prompt, continuation)
        
        # Check for scientific structure
        structure_score = 0.0
        
        # Check for causal connectives
        causal_connectives = ['therefore', 'thus', 'consequently', 'as a result', 'because']
        if any(conn in continuation.lower() for conn in causal_connectives):
            structure_score += 0.3
        
        # Check for scientific units
        if self.unit_pattern.search(continuation):
            structure_score += 0.2
        
        # Check for formulas
        if self.formula_pattern.search(continuation):
            structure_score += 0.2
        
        # Check for field-specific terms
        if field == "chemistry":
            chem_terms = ['reaction', 'compound', 'molecule', 'element', 'bond']
            if any(term in continuation.lower() for term in chem_terms):
                structure_score += 0.3
        elif field == "physics":
            phys_terms = ['force', 'energy', 'momentum', 'field', 'particle']
            if any(term in continuation.lower() for term in phys_terms):
                structure_score += 0.3
        
        # Combine scores
        coherence = rouge_score * 0.5 + min(structure_score, 1.0) * 0.5
        
        return coherence
    
    def calculate_rouge_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate ROUGE-L similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            ROUGE-L F1 score
        """
        scores = self.rouge_scorer.score(text1, text2)
        return scores['rougeL'].fmeasure
    
    def build_candidate_set(self, 
                          prompt_data: Dict, 
                          responses: List[str], 
                          max_set_size: int = 32, 
                          ppl_threshold_percentile: int = 50, 
                          rouge_threshold: float = 0.3,
                          coherence_threshold: float = 0.4) -> List[Dict]:
        """
        Build high-quality candidate set C(x) for scientific text
        
        Args:
            prompt_data: Prompt metadata including field
            responses: List of generated responses
            max_set_size: Maximum size of candidate set
            ppl_threshold_percentile: Percentile threshold for perplexity filtering
            rouge_threshold: Maximum ROUGE-L similarity threshold for diversity
            coherence_threshold: Minimum scientific coherence threshold
            
        Returns:
            Candidate set with quality metrics
        """
        prompt_text = prompt_data['prompt']
        field = prompt_data.get('field', 'general')
        
        # Calculate metrics for all responses
        response_data = []
        for idx, response in enumerate(responses):
            full_text = prompt_text + " " + response
            
            # Calculate perplexity
            ppl = self.calculate_perplexity(full_text)
            
            # Calculate scientific coherence
            coherence = self.calculate_scientific_coherence(prompt_text, response, field)
            
            # Extract features
            has_formula = bool(self.formula_pattern.search(response))
            has_units = bool(self.unit_pattern.search(response))
            
            response_info = {
                'idx': idx,
                'text': response,
                'perplexity': ppl,
                'coherence': coherence,
                'has_formula': has_formula,
                'has_units': has_units,
                'rouge_scores': [],
                'field': field
            }
            response_data.append(response_info)
        
        # Filter by perplexity (lower is better)
        ppls = [r['perplexity'] for r in response_data]
        ppl_threshold = np.percentile(ppls, ppl_threshold_percentile)
        filtered_responses = [r for r in response_data if r['perplexity'] <= ppl_threshold]
        
        # Filter by coherence
        filtered_responses = [r for r in filtered_responses if r['coherence'] >= coherence_threshold]
        
        # If too few responses after filtering, relax coherence threshold
        if len(filtered_responses) < max_set_size // 2:
            filtered_responses = [r for r in response_data if r['perplexity'] <= ppl_threshold]
        
        # Calculate pairwise ROUGE-L similarities
        for i, resp1 in enumerate(filtered_responses):
            for j, resp2 in enumerate(filtered_responses):
                if i != j:
                    rouge_score = self.calculate_rouge_similarity(resp1['text'], resp2['text'])
                    resp1['rouge_scores'].append(rouge_score)
        
        # Select diverse responses with low similarity
        candidate_set = []
        
        # Sort by combined score (lower perplexity + higher coherence)
        filtered_responses.sort(key=lambda x: x['perplexity'] - x['coherence'] * 100)
        
        # Prioritize responses with formulas/units for science
        formula_responses = [r for r in filtered_responses if r['has_formula'] or r['has_units']]
        other_responses = [r for r in filtered_responses if not (r['has_formula'] or r['has_units'])]
        
        # Reorder: formula responses first
        ordered_responses = formula_responses + other_responses
        
        for response in ordered_responses:
            if len(candidate_set) >= max_set_size:
                break
            
            # Check if this response is diverse enough
            is_diverse = True
            for selected in candidate_set:
                similarity = self.calculate_rouge_similarity(response['text'], selected['text'])
                if similarity > rouge_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                candidate_set.append(response)
        
        # Log statistics
        logger.info(f"Built candidate set: {len(candidate_set)} responses "
                   f"(from {len(responses)} generated, {len(filtered_responses)} after quality filter)")
        
        return candidate_set
    
    def process_dataset(self, 
                       prompts: List[Dict], 
                       output_dir: str = "./processed_science_data", 
                       num_responses_per_prompt: int = 40, 
                       batch_size: int = 10) -> str:
        """
        Process entire dataset: generate responses and build candidate sets
        
        Args:
            prompts: List of prompts to process
            output_dir: Directory to save processed data
            num_responses_per_prompt: Number of responses to generate per prompt
            batch_size: Batch size for saving intermediate results
            
        Returns:
            Path to saved processed data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        processed_data = {}
        
        for i, prompt_data in enumerate(tqdm(prompts, desc="Processing science prompts")):
            # Generate responses with field awareness
            responses = self.generate_responses(
                prompt_data['prompt'], 
                field=prompt_data.get('field', 'general'),
                num_responses=num_responses_per_prompt
            )
            
            # Build candidate set with scientific quality metrics
            candidate_set = self.build_candidate_set(prompt_data, responses)
            
            # Store processed data
            processed_data[prompt_data['idx']] = {
                'prompt': prompt_data,
                'all_responses': responses,
                'candidate_set': candidate_set,
                'candidate_indices': [c['idx'] for c in candidate_set],
                'field': prompt_data.get('field', 'general'),
                'source': prompt_data.get('source', 'unknown')
            }
            
            # Save intermediate results
            if (i + 1) % batch_size == 0:
                batch_file = os.path.join(output_dir, f'batch_{i//batch_size}.pkl')
                with open(batch_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                logger.info(f"Saved batch {i//batch_size}")
        
        # Save final results
        final_file = os.path.join(output_dir, 'processed_science_data_complete.pkl')
        with open(final_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Also save summary statistics
        self._save_processing_stats(processed_data, output_dir)
        
        logger.info(f"Processing complete. Saved to {final_file}")
        return final_file
    
    def _save_processing_stats(self, processed_data: Dict, output_dir: str):
        """Save statistics about the processed dataset"""
        stats = {
            'total_prompts': len(processed_data),
            'avg_candidate_set_size': np.mean([len(d['candidate_set']) for d in processed_data.values()]),
            'avg_perplexity': [],
            'avg_coherence': [],
            'field_distribution': defaultdict(int),
            'source_distribution': defaultdict(int),
            'formulas_count': 0,
            'units_count': 0
        }
        
        for data in processed_data.values():
            stats['field_distribution'][data['field']] += 1
            stats['source_distribution'][data['source']] += 1
            
            for response in data['candidate_set']:
                stats['avg_perplexity'].append(response['perplexity'])
                stats['avg_coherence'].append(response['coherence'])
                if response['has_formula']:
                    stats['formulas_count'] += 1
                if response['has_units']:
                    stats['units_count'] += 1
        
        stats['avg_perplexity'] = np.mean(stats['avg_perplexity'])
        stats['avg_coherence'] = np.mean(stats['avg_coherence'])
        
        # Save stats
        stats_file = os.path.join(output_dir, 'processing_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset statistics saved to {stats_file}")
    
    def prepare_calibration_data(self, processed_data_path: str, 
                               science_scores_path: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Prepare calibration data in the format expected by sampling.py
        
        Args:
            processed_data_path: Path to processed data file
            science_scores_path: Path to pre-computed science scores (if available)
            
        Returns:
            x_cal and y_cal dictionaries
        """
        with open(processed_data_path, 'rb') as f:
            processed_data = pickle.load(f)
        
        x_cal = {}
        y_cal = defaultdict(dict)
        
        # Initialize the 'set' key for y_cal[0]
        y_cal[0]['set'] = {}
        
        for idx, data in processed_data.items():
            # Prepare x_cal entry
            x_cal[idx] = {
                'pred': [resp['text'] for resp in data['candidate_set']],
                'science_validity_ft': None,  # To be filled by science scoring module
                'science_validity_human': None,  # To be filled by human evaluation
                'field': data['field'],
                'prompt': data['prompt']['prompt'],
                'metadata': {
                    'perplexities': [resp['perplexity'] for resp in data['candidate_set']],
                    'coherences': [resp['coherence'] for resp in data['candidate_set']],
                    'has_formulas': [resp['has_formula'] for resp in data['candidate_set']],
                    'has_units': [resp['has_units'] for resp in data['candidate_set']]
                }
            }
            
            # Prepare y_cal entry (candidate set with indices)
            y_cal[0]['set'][idx] = [(i, resp['text']) for i, resp in enumerate(data['candidate_set'])]
        
        # Load pre-computed scores if available
        if science_scores_path and os.path.exists(science_scores_path):
            with open(science_scores_path, 'rb') as f:
                science_scores = pickle.load(f)
            
            for idx in x_cal:
                if idx in science_scores:
                    x_cal[idx]['science_validity_ft'] = science_scores[idx].get('machine_scores')
                    x_cal[idx]['science_validity_human'] = science_scores[idx].get('human_scores')
        
        return x_cal, y_cal
    
    def create_human_evaluation_template(self, x_cal: Dict, output_file: str):
        """
        Create template for human evaluation of scientific validity
        
        Args:
            x_cal: Calibration data
            output_file: Output file path for evaluation template
        """
        evaluation_data = []
        
        for idx, data in x_cal.items():
            for i, text in enumerate(data['pred']):
                eval_item = {
                    'prompt_id': idx,
                    'response_id': i,
                    'field': data['field'],
                    'prompt': data['prompt'],
                    'response': text,
                    'science_validity_score': None,  # To be filled by human
                    'violations': [],  # List of scientific violations if any
                    'notes': ""
                }
                evaluation_data.append(eval_item)
        
        # Save as CSV for easy annotation
        df = pd.DataFrame(evaluation_data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Created human evaluation template: {output_file}")
        logger.info(f"Total items for evaluation: {len(evaluation_data)}")


# Backward compatibility wrapper
class DataPreprocessor(ScienceDataPreprocessor):
    """Wrapper for backward compatibility with original interface"""
    
    def load_realtoxicityprompts(self, split="train", num_samples=None):
        """Load science prompts instead of toxicity prompts"""
        # Map to science prompts
        data_file = "./processed_science_data/science_prompts.pkl"
        if not os.path.exists(data_file):
            logger.warning(f"Science prompts file not found at {data_file}. "
                         f"Please run download_science_data.py first.")
            return []
        
        return self.load_science_prompts(data_file, num_samples)


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ScienceDataPreprocessor(
        model_name="facebook/galactica-1.3b",  # Or "facebook/galactica-6.7b" for better quality
        use_multiple_models=False  # Set True for more diversity
    )
    
    # Load science prompts
    prompts = preprocessor.load_science_prompts(
        "./processed_science_data/science_prompts_test.pkl",
        num_samples=10  # Start with small number for testing
    )
    
    # Process dataset
    processed_file = preprocessor.process_dataset(prompts)
    
    # Prepare calibration data
    x_cal, y_cal = preprocessor.prepare_calibration_data(processed_file)
    
    # Create human evaluation template
    preprocessor.create_human_evaluation_template(
        x_cal, 
        "./processed_science_data/human_evaluation_template.csv"
    )
    
    print(f"Processed {len(x_cal)} prompts")
    print(f"Average candidate set size: {np.mean([len(v['pred']) for v in x_cal.values()]):.2f}")
    
    # Show sample
    sample_idx = list(x_cal.keys())[0]
    print(f"\nSample prompt: {x_cal[sample_idx]['prompt']}")
    print(f"Field: {x_cal[sample_idx]['field']}")
    print(f"Number of candidates: {len(x_cal[sample_idx]['pred'])}")
    print(f"First candidate: {x_cal[sample_idx]['pred'][0][:100]}...")