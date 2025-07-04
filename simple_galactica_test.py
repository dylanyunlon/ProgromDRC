"""
Simple test for Galactica model generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_galactica_generation():
    """Test basic Galactica generation"""
    
    print("Loading Galactica model and tokenizer...")
    
    # Load tokenizer and model
    model_name = "facebook/galactica-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure tokenizer
    print(f"Original pad_token: {tokenizer.pad_token}")
    print(f"Original eos_token: {tokenizer.eos_token}")
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added pad_token: {tokenizer.pad_token}")
    
    # Test prompt
    prompt = "When water freezes, it"
    print(f"\nPrompt: {prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Remove token_type_ids if present (Galactica doesn't use them)
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
        print("Removed token_type_ids")
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nGenerated: {generated_text}")
    
    # Extract only the continuation
    continuation = generated_text[len(prompt):].strip()
    print(f"\nContinuation: {continuation}")
    
    print("\n✓ Test successful!")


def test_batch_generation():
    """Test batch generation with Galactica"""
    
    print("\n" + "="*50)
    print("Testing batch generation...")
    
    model_name = "facebook/galactica-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
    
    # Multiple prompts
    prompts = [
        "The chemical formula for water is",
        "According to Newton's first law,",
        "When temperature increases, the reaction rate"
    ]
    
    print("\nPrompts:")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. {p}")
    
    # Tokenize batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    
    # Remove token_type_ids
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode all
    print("\nResults:")
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        generated = tokenizer.decode(output, skip_special_tokens=True)
        continuation = generated[len(prompt):].strip()
        print(f"\n{i+1}. Prompt: {prompt}")
        print(f"   Continuation: {continuation}")
    
    print("\n✓ Batch test successful!")


if __name__ == "__main__":
    try:
        # Test single generation
        test_galactica_generation()
        
        # Test batch generation
        test_batch_generation()
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
