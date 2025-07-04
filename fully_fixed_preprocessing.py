#!/usr/bin/env python
"""
Fully fixed preprocessing script for Galactica
Handles all token_type_ids issues
"""

import os
import pickle
import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FullyFixedGalacticaPreprocessor:
    """Complete preprocessor with all Galactica fixes"""
    
    def __init__(self, 
                 model_name="facebook/galactica-1.3b",
                 device="cuda",
                 cache_dir="./cache"):
        """Initialize with complete Galactica compatibility"""
        
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Loading {model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Fix padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            logger.info("Added <pad> token")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Resize embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Science patterns
        self.formula_pattern = re.compile(r'(\$[^\$]+\$|\\\[[^\]]+\\\]|[A-Z][a-z]?\d*(?:\s*[+-]\s*[A-Z][a-z]?\d*)*)')
        self.unit_pattern = re.compile(r'\d+\s*(K|°C|°F|m/s|N|J|mol|M|mL|L|g|kg|Pa|atm)')
        
        logger.info("Initialization complete!")
    
    def _prepare_inputs(self, text, **kwargs):
        """Prepare inputs with token_type_ids removed"""
        inputs = self.tokenizer(text, return_tensors="pt", **kwargs)
        
        # Always remove token_type_ids for Galactica
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def generate_responses(self, 
                         prompt_text: str, 
                         field: str = "general",
                         num_responses: int = 40, 
                         max_new_tokens: int = 100, 
                         temperature: float = 0.8, 
                         top_p: float = 0.9) -> list:
        """Generate responses with full compatibility"""
        
        # Add field guidance
        if field == "chemistry":
            guided_prompt = f"[Chemistry] {prompt_text}"
        elif field == "physics":
            guided_prompt = f"[Physics] {prompt_text}"
        else:
            guided_prompt = prompt_text
        
        # Prepare inputs
        inputs = self._prepare_inputs(guided_prompt, padding=True)
        
        responses = []
        
        # Generate
        with torch.no_grad():
            for _ in range(num_responses):
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                responses.append(generated_text)
        
        # Clean responses
        cleaned_responses = []
        for response in responses:
            # Remove field guidance
            if field != "general":
                response = response.replace(f"[{field.capitalize()}] ", "")
            
            # Remove prompt
            if response.startswith(prompt_text):
                response = response[len(prompt_text):].strip()
            
            cleaned_responses.append(response)
        
        return cleaned_responses
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity with compatibility"""
        inputs = self._prepare_inputs(text, padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            # Use inputs['input_ids'] instead of inputs.input_ids
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def calculate_rouge_similarity(self, text1: str, text2: str) -> float:
        """Calculate ROUGE-L similarity"""
        scores = self.rouge_scorer.score(text1, text2)
        return scores['rougeL'].fmeasure
    
    def calculate_scientific_coherence(self, prompt: str, continuation: str, field: str) -> float:
        """Calculate scientific coherence score"""
        # Basic ROUGE similarity
        rouge_score = self.calculate_rouge_similarity(prompt, continuation)
        
        # Check for scientific structure
        structure_score = 0.0
        
        # Causal connectives
        causal_connectives = ['therefore', 'thus', 'consequently', 'as a result', 'because']
        if any(conn in continuation.lower() for conn in causal_connectives):
            structure_score += 0.3
        
        # Scientific units
        if self.unit_pattern.search(continuation):
            structure_score += 0.2
        
        # Formulas
        if self.formula_pattern.search(continuation):
            structure_score += 0.2
        
        # Field-specific terms
        if field == "chemistry":
            chem_terms = ['reaction', 'compound', 'molecule', 'element', 'bond']
            if any(term in continuation.lower() for term in chem_terms):
                structure_score += 0.3
        elif field == "physics":
            phys_terms = ['force', 'energy', 'momentum', 'field', 'particle']
            if any(term in continuation.lower() for term in phys_terms):
                structure_score += 0.3
        
        coherence = rouge_score * 0.5 + min(structure_score, 1.0) * 0.5
        return coherence
    
    def build_candidate_set(self, 
                          prompt_data: dict, 
                          responses: list, 
                          max_set_size: int = 32, 
                          ppl_threshold_percentile: int = 50, 
                          rouge_threshold: float = 0.3,
                          coherence_threshold: float = 0.4) -> list:
        """Build candidate set with quality filtering"""
        
        prompt_text = prompt_data['prompt']
        field = prompt_data.get('field', 'general')
        
        # Calculate metrics for all responses
        response_data = []
        
        logger.info(f"Calculating metrics for {len(responses)} responses...")
        
        for idx, response in enumerate(responses):
            full_text = prompt_text + " " + response
            
            # Calculate perplexity
            try:
                ppl = self.calculate_perplexity(full_text)
            except Exception as e:
                logger.warning(f"Perplexity calculation failed: {e}")
                ppl = 1000.0  # High penalty
            
            # Calculate coherence
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
                'field': field
            }
            response_data.append(response_info)
        
        # Filter by perplexity
        ppls = [r['perplexity'] for r in response_data]
        ppl_threshold = np.percentile(ppls, ppl_threshold_percentile)
        filtered_responses = [r for r in response_data if r['perplexity'] <= ppl_threshold]
        
        # Filter by coherence
        filtered_responses = [r for r in filtered_responses if r['coherence'] >= coherence_threshold]
        
        # If too few, relax coherence
        if len(filtered_responses) < max_set_size // 2:
            filtered_responses = [r for r in response_data if r['perplexity'] <= ppl_threshold]
        
        # Select diverse responses
        candidate_set = []
        
        # Sort by combined score
        filtered_responses.sort(key=lambda x: x['perplexity'] - x['coherence'] * 100)
        
        # Prioritize responses with formulas/units
        formula_responses = [r for r in filtered_responses if r['has_formula'] or r['has_units']]
        other_responses = [r for r in filtered_responses if not (r['has_formula'] or r['has_units'])]
        ordered_responses = formula_responses + other_responses
        
        for response in ordered_responses:
            if len(candidate_set) >= max_set_size:
                break
            
            # Check diversity
            is_diverse = True
            for selected in candidate_set:
                similarity = self.calculate_rouge_similarity(response['text'], selected['text'])
                if similarity > rouge_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                candidate_set.append(response)
        
        logger.info(f"Built candidate set: {len(candidate_set)} responses")
        
        return candidate_set
    
    def load_science_prompts(self, data_file: str, num_samples: int = None) -> list:
        """Load science prompts"""
        with open(data_file, 'rb') as f:
            prompts = pickle.load(f)
        
        if num_samples:
            prompts = prompts[:num_samples]
        
        logger.info(f"Loaded {len(prompts)} prompts")
        return prompts
    
    def process_dataset(self, prompts: list, output_dir: str, 
                       num_responses_per_prompt: int = 40) -> str:
        """Process entire dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        processed_data = {}
        
        for i, prompt_data in enumerate(tqdm(prompts, desc="Processing prompts")):
            try:
                # Generate responses
                responses = self.generate_responses(
                    prompt_data['prompt'],
                    field=prompt_data.get('field', 'general'),
                    num_responses=num_responses_per_prompt
                )
                
                # Build candidate set
                candidate_set = self.build_candidate_set(prompt_data, responses)
                
                # Store data
                processed_data[prompt_data['idx']] = {
                    'prompt': prompt_data,
                    'all_responses': responses,
                    'candidate_set': candidate_set,
                    'candidate_indices': [c['idx'] for c in candidate_set],
                    'field': prompt_data.get('field', 'general'),
                    'source': prompt_data.get('source', 'unknown')
                }
                
                # Save batch
                if (i + 1) % 5 == 0:
                    batch_file = os.path.join(output_dir, f'batch_{i//5}.pkl')
                    with open(batch_file, 'wb') as f:
                        pickle.dump(processed_data, f)
                
            except Exception as e:
                logger.error(f"Error processing prompt {i}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save final
        final_file = os.path.join(output_dir, 'processed_science_data_complete.pkl')
        with open(final_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        return final_file
    
    def prepare_calibration_data(self, processed_data_path: str) -> tuple:
        """Prepare calibration data"""
        with open(processed_data_path, 'rb') as f:
            processed_data = pickle.load(f)
        
        x_cal = {}
        y_cal = defaultdict(dict)
        
        # Initialize the 'set' key for y_cal[0]
        y_cal[0]['set'] = {}
        
        for idx, data in processed_data.items():
            x_cal[idx] = {
                'pred': [resp['text'] for resp in data['candidate_set']],
                'field': data['field'],
                'prompt': data['prompt']['prompt'],
                'metadata': {
                    'perplexities': [resp['perplexity'] for resp in data['candidate_set']],
                    'coherences': [resp['coherence'] for resp in data['candidate_set']]
                }
            }
            
            y_cal[0]['set'][idx] = [(i, resp['text']) for i, resp in enumerate(data['candidate_set'])]
        
        return x_cal, y_cal


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-prompts', type=int, default=10)
    parser.add_argument('--num-responses', type=int, default=10)
    parser.add_argument('--prompts-file', default='./processed_science_data/science_prompts.pkl')
    parser.add_argument('--output-dir', default='./processed_science_data')
    
    args = parser.parse_args()
    
    # Initialize
    preprocessor = FullyFixedGalacticaPreprocessor()
    
    # Load prompts
    prompts = preprocessor.load_science_prompts(args.prompts_file, args.num_prompts)
    
    # Quick test
    logger.info("Running quick test...")
    test_responses = preprocessor.generate_responses(
        prompts[0]['prompt'],
        field=prompts[0].get('field', 'general'),
        num_responses=2,
        max_new_tokens=50
    )
    logger.info(f"Test successful: {test_responses[0][:100]}...")
    
    # Process
    logger.info("Processing dataset...")
    output_file = preprocessor.process_dataset(
        prompts,
        args.output_dir,
        num_responses_per_prompt=args.num_responses
    )
    
    logger.info(f"Saved to: {output_file}")
    
    # Prepare calibration
    x_cal, y_cal = preprocessor.prepare_calibration_data(output_file)
    
    cal_file = os.path.join(args.output_dir, 'calibration_data.pkl')
    with open(cal_file, 'wb') as f:
        pickle.dump({'x_cal': x_cal, 'y_cal': y_cal}, f)
    
    logger.info(f"Calibration data saved to: {cal_file}")
    logger.info("✓ Complete!")


if __name__ == "__main__":
    main()