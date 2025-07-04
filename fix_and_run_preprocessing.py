"""
Fix and run data preprocessing with proper tokenizer configuration
"""
from typing import List
import os
import sys
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import ScienceDataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedGalacticaPreprocessor(ScienceDataPreprocessor):
    """Fixed version of ScienceDataPreprocessor for Galactica models"""
    
    def __init__(self, 
                 model_name="facebook/galactica-1.3b",
                 device="cuda", 
                 cache_dir="./cache",
                 use_multiple_models=False):
        """Initialize with fixed tokenizer configuration"""
        
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.use_multiple_models = use_multiple_models
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize primary model
        logger.info(f"Loading primary model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Fix for Galactica tokenizer
        if "galactica" in model_name.lower():
            logger.info("Detected Galactica model, configuring tokenizer...")
            # Galactica needs special handling
            if self.tokenizer.pad_token is None:
                # Add pad token for Galactica
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
                logger.info(f"Added pad_token: {self.tokenizer.pad_token} (id: {self.tokenizer.pad_token_id})")
        elif self.tokenizer.pad_token is None:
            # For other models without pad token
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True  # For Galactica
        )
        
        # Resize token embeddings if we added new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize additional components
        if use_multiple_models:
            self._load_additional_models()
        
        # Initialize ROUGE scorer
        from rouge_score import rouge_scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Science-specific patterns
        import re
        self.formula_pattern = re.compile(r'(\$[^\$]+\$|\\\[[^\]]+\\\]|[A-Z][a-z]?\d*(?:\s*[+-]\s*[A-Z][a-z]?\d*)*)')
        self.unit_pattern = re.compile(r'\d+\s*(K|°C|°F|m/s|N|J|mol|M|mL|L|g|kg|Pa|atm)')
        
        logger.info("Preprocessor initialization complete!")
    
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
        inputs = self.tokenizer(guided_prompt, return_tensors="pt", padding=True)
        
        # Remove token_type_ids for Galactica
        if 'token_type_ids' in inputs and "galactica" in self.model_name.lower():
            inputs.pop('token_type_ids')
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        
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
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                responses.append(generated_text)
        
        # Clean responses
        cleaned_responses = []
        for response in responses:
            # Remove field guidance if added
            if use_field_guidance and field != "general":
                response = response.replace(f"[{field.capitalize()}] ", "")
            
            # Remove original prompt
            if response.startswith(prompt_text):
                response = response[len(prompt_text):].strip()
            
            cleaned_responses.append(response)
        
        return cleaned_responses[:num_responses]


def main():
    """Main function to run preprocessing with fixed configuration"""
    
    # Check if science prompts file exists
    prompts_file = './processed_science_data/science_prompts.pkl'
    if not os.path.exists(prompts_file):
        logger.error(f"Science prompts file not found: {prompts_file}")
        logger.info("Please run download_science_data.py first to generate the prompts file.")
        return
    
    try:
        # Initialize fixed preprocessor
        logger.info("Initializing preprocessor with fixed tokenizer configuration...")
        preprocessor = FixedGalacticaPreprocessor(
            model_name="facebook/galactica-1.3b",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load prompts
        logger.info("Loading science prompts...")
        prompts = preprocessor.load_science_prompts(prompts_file, num_samples=100)
        
        # Process dataset
        logger.info("Processing dataset...")
        output_file = preprocessor.process_dataset(
            prompts,
            output_dir="./processed_science_data",
            num_responses_per_prompt=40,
            batch_size=10
        )
        
        logger.info(f"Processing complete! Output saved to: {output_file}")
        
        # Prepare calibration data
        logger.info("Preparing calibration data...")
        x_cal, y_cal = preprocessor.prepare_calibration_data(output_file)
        
        logger.info(f"Calibration data prepared:")
        logger.info(f"  - Number of prompts: {len(x_cal)}")
        logger.info(f"  - Average candidate set size: {sum(len(v['pred']) for v in x_cal.values()) / len(x_cal):.2f}")
        
        # Create human evaluation template
        eval_template_file = "./processed_science_data/human_evaluation_template.csv"
        preprocessor.create_human_evaluation_template(x_cal, eval_template_file)
        logger.info(f"Human evaluation template saved to: {eval_template_file}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


def test_quick():
    """Quick test with a single prompt"""
    
    logger.info("Running quick test...")
    
    try:
        import torch
        
        # Initialize preprocessor
        preprocessor = FixedGalacticaPreprocessor(
            model_name="facebook/galactica-1.3b",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Test with a single prompt
        test_prompt = "When temperature increases in a chemical reaction,"
        logger.info(f"Test prompt: {test_prompt}")
        
        responses = preprocessor.generate_responses(
            test_prompt,
            field="chemistry",
            num_responses=3,
            max_new_tokens=50
        )
        
        logger.info("Generated responses:")
        for i, response in enumerate(responses):
            logger.info(f"  {i+1}: {response}")
        
        logger.info("✓ Quick test passed!")
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description="Run science data preprocessing with fixed configuration")
    parser.add_argument("--test", action="store_true", help="Run quick test only")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of prompts to process")
    
    args = parser.parse_args()
    
    if args.test:
        test_quick()
    else:
        # Run full preprocessing
        main()