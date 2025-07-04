def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity with Galactica compatibility"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Remove token_type_ids
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity#!/usr/bin/env python
"""
Run science data preprocessing with proper Galactica configuration
Based on successful test results
"""

import os
import sys
import pickle
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

# Import the modified data preprocessing module
from data_preprocessing import ScienceDataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GalacticaSciencePreprocessor(ScienceDataPreprocessor):
    """Override to ensure Galactica compatibility"""
    
    def __init__(self, *args, **kwargs):
        """Initialize with Galactica-specific fixes"""
        # Don't call parent __init__ yet
        self.model_name = kwargs.get('model_name', "facebook/galactica-1.3b")
        self.device = kwargs.get('device', 'cuda')
        self.cache_dir = kwargs.get('cache_dir', './cache')
        self.use_multiple_models = kwargs.get('use_multiple_models', False)
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize Galactica with proper configuration
        logger.info(f"Loading Galactica model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        
        # Fix padding token (as proven in the test)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            logger.info(f"Added pad_token: {self.tokenizer.pad_token}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Resize embeddings if we added tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize other components
        from rouge_score import rouge_scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        import re
        self.formula_pattern = re.compile(r'(\$[^\$]+\$|\\\[[^\]]+\\\]|[A-Z][a-z]?\d*(?:\s*[+-]\s*[A-Z][a-z]?\d*)*)')
        self.unit_pattern = re.compile(r'\d+\s*(K|°C|°F|m/s|N|J|mol|M|mL|L|g|kg|Pa|atm)')
        
        logger.info("Galactica preprocessor initialized successfully!")
    
    def generate_responses(self, 
                         prompt_text: str, 
                         field: str = "general",
                         num_responses: int = 40, 
                         max_new_tokens: int = 100, 
                         temperature: float = 0.8, 
                         top_p: float = 0.9,
                         use_field_guidance: bool = True) -> list:
        """Generate responses with Galactica-specific handling"""
        
        # Add field guidance if requested
        if use_field_guidance and field != "general":
            if field == "chemistry":
                guided_prompt = f"[Chemistry] {prompt_text}"
            elif field == "physics":
                guided_prompt = f"[Physics] {prompt_text}"
            else:
                guided_prompt = prompt_text
        else:
            guided_prompt = prompt_text
        
        # Tokenize
        inputs = self.tokenizer(guided_prompt, return_tensors="pt", padding=True)
        
        # Remove token_type_ids (critical for Galactica/OPT)
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        responses = []
        
        # Generate responses
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
            if use_field_guidance and field != "general":
                response = response.replace(f"[{field.capitalize()}] ", "")
            
            # Remove original prompt
            if response.startswith(prompt_text):
                response = response[len(prompt_text):].strip()
            
            cleaned_responses.append(response)
        
        return cleaned_responses[:num_responses]


def main():
    """Main preprocessing function"""
    
    # Configuration
    prompts_file = './processed_science_data/science_prompts.pkl'
    output_dir = './processed_science_data'
    num_prompts = 100  # Adjust as needed
    num_responses_per_prompt = 40
    
    # Check if prompts file exists
    if not os.path.exists(prompts_file):
        logger.error(f"Prompts file not found: {prompts_file}")
        logger.info("Please run download_science_data.py first!")
        return
    
    # Load prompts
    logger.info(f"Loading prompts from {prompts_file}")
    with open(prompts_file, 'rb') as f:
        all_prompts = pickle.load(f)
    
    # Use subset
    prompts = all_prompts[:num_prompts]
    logger.info(f"Processing {len(prompts)} prompts")
    
    # Initialize preprocessor
    logger.info("Initializing Galactica preprocessor...")
    preprocessor = GalacticaSciencePreprocessor(
        model_name="facebook/galactica-1.3b",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Quick test
    logger.info("Running quick test...")
    test_prompt = prompts[0]
    test_responses = preprocessor.generate_responses(
        test_prompt['prompt'],
        field=test_prompt.get('field', 'general'),
        num_responses=2,
        max_new_tokens=50
    )
    logger.info(f"Test prompt: {test_prompt['prompt']}")
    logger.info(f"Test response 1: {test_responses[0][:100]}...")
    
    # Process full dataset
    logger.info("Processing full dataset...")
    processed_data = {}
    
    for i, prompt_data in enumerate(tqdm(prompts, desc="Processing prompts")):
        try:
            # Generate responses
            responses = preprocessor.generate_responses(
                prompt_data['prompt'],
                field=prompt_data.get('field', 'general'),
                num_responses=num_responses_per_prompt
            )
            
            # Build candidate set
            candidate_set = preprocessor.build_candidate_set(prompt_data, responses)
            
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
            if (i + 1) % 10 == 0:
                batch_file = os.path.join(output_dir, f'batch_{i//10}.pkl')
                with open(batch_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                logger.info(f"Saved batch {i//10}")
                
        except Exception as e:
            logger.error(f"Error processing prompt {i}: {e}")
            continue
    
    # Save final results
    final_file = os.path.join(output_dir, 'processed_science_data_complete.pkl')
    with open(final_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    logger.info(f"Processing complete! Saved to {final_file}")
    
    # Prepare calibration data
    logger.info("Preparing calibration data...")
    x_cal, y_cal = preprocessor.prepare_calibration_data(final_file)
    
    # Save calibration data
    cal_file = os.path.join(output_dir, 'calibration_data.pkl')
    with open(cal_file, 'wb') as f:
        pickle.dump({'x_cal': x_cal, 'y_cal': y_cal}, f)
    
    logger.info(f"Calibration data saved to {cal_file}")
    
    # Print statistics
    logger.info("\nProcessing Statistics:")
    logger.info(f"Total prompts processed: {len(processed_data)}")
    logger.info(f"Average candidate set size: {np.mean([len(d['candidate_set']) for d in processed_data.values()]):.2f}")
    
    # Field distribution
    fields = [d['field'] for d in processed_data.values()]
    field_counts = {f: fields.count(f) for f in set(fields)}
    logger.info(f"Field distribution: {field_counts}")
    
    # Create human evaluation template
    logger.info("Creating human evaluation template...")
    eval_file = os.path.join(output_dir, 'human_evaluation_template.csv')
    preprocessor.create_human_evaluation_template(x_cal, eval_file)
    logger.info(f"Evaluation template saved to {eval_file}")
    
    logger.info("\n✓ All processing complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-prompts', type=int, default=100, help='Number of prompts to process')
    parser.add_argument('--num-responses', type=int, default=40, help='Responses per prompt')
    parser.add_argument('--test-only', action='store_true', help='Run test only')
    
    args = parser.parse_args()
    
    if args.test_only:
        # Just test with a few prompts
        logger.info("Running test mode...")
        # Modify the main function parameters here if needed
    
    main()