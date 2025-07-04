"""
Test script to verify Galactica tokenizer configuration
"""

from transformers import AutoTokenizer
import torch

def test_galactica_tokenizer():
    """Test and fix Galactica tokenizer padding configuration"""
    
    print("Loading Galactica tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
    
    print(f"Initial pad_token: {tokenizer.pad_token}")
    print(f"Initial pad_token_id: {tokenizer.pad_token_id}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"EOS token_id: {tokenizer.eos_token_id}")
    
    # Fix padding token
    if tokenizer.pad_token is None:
        print("\nPadding token not set. Setting it now...")
        
        # For Galactica, the pad token should be set to eos_token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        else:
            # Fallback: add a new pad token
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            print(f"Added new pad_token: {tokenizer.pad_token}")
    
    # Test tokenization with padding
    test_texts = [
        "When temperature increases,",
        "In a chemical reaction, the rate of reaction depends on temperature."
    ]
    
    print("\nTesting tokenization with padding...")
    try:
        # Test single text
        single_encoded = tokenizer(test_texts[0], return_tensors="pt", padding=True)
        print("✓ Single text tokenization successful")
        
        # Test batch
        batch_encoded = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=50)
        print("✓ Batch tokenization successful")
        print(f"  Input IDs shape: {batch_encoded['input_ids'].shape}")
        print(f"  Attention mask shape: {batch_encoded['attention_mask'].shape}")
        
        # Decode to verify
        decoded = tokenizer.batch_decode(batch_encoded['input_ids'], skip_special_tokens=True)
        print("\nDecoded texts:")
        for i, text in enumerate(decoded):
            print(f"  {i}: {text}")
            
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True


def create_fixed_preprocessor():
    """Create a data preprocessor with fixed tokenizer configuration"""
    from data_preprocessing import ScienceDataPreprocessor
    
    class FixedScienceDataPreprocessor(ScienceDataPreprocessor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Ensure padding token is properly set for Galactica
            if "galactica" in self.model_name.lower():
                if self.tokenizer.pad_token is None:
                    # Galactica uses </s> as both eos and pad token
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    print(f"Fixed Galactica padding token: {self.tokenizer.pad_token}")
    
    return FixedScienceDataPreprocessor


if __name__ == "__main__":
    # Test tokenizer
    test_galactica_tokenizer()
    
    # Test with fixed preprocessor
    print("\n" + "="*50)
    print("Testing with fixed preprocessor...")
    
    PreprocessorClass = create_fixed_preprocessor()
    preprocessor = PreprocessorClass()
    
    # Test generation
    test_prompt = "When water freezes,"
    print(f"\nTest prompt: {test_prompt}")
    
    try:
        responses = preprocessor.generate_responses(
            test_prompt,
            field="physics",
            num_responses=2,
            max_new_tokens=30
        )
        
        print("Generated responses:")
        for i, response in enumerate(responses):
            print(f"  {i+1}: {response}")
            
        print("\n✓ Generation successful!")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
