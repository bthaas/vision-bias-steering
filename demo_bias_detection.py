#!/usr/bin/env python3
"""
Demo script to show how gender bias detection works in the model.
This will show step-by-step how the system measures bias.
"""

import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from bias_steering.steering import load_model
from bias_steering.data.load_dataset import load_target_words, load_gendered_language_dataset
from bias_steering.steering.steering_utils import get_target_token_ids

def demo_bias_detection():
    print("=" * 70)
    print("GENDER BIAS DETECTION DEMO")
    print("=" * 70)
    print()
    
    # Load model
    print("1. Loading model (GPT-2)...")
    model_name = "gpt2"
    model = load_model(model_name)
    print(f"   ✓ Model loaded: {model_name}")
    print()
    
    # Load target words
    print("2. Loading target words for gender bias detection...")
    target_words = load_target_words(target_concept="gender")
    print(f"   Female-related words: {target_words['F'][:5]}... (and {len(target_words['F'])-5} more)")
    print(f"   Male-related words: {target_words['M'][:5]}... (and {len(target_words['M'])-5} more)")
    print()
    
    # Get token IDs
    print("3. Converting words to token IDs...")
    target_token_ids = {
        "F": get_target_token_ids(model.tokenizer, target_words["F"]),
        "M": get_target_token_ids(model.tokenizer, target_words["M"])
    }
    print(f"   Female token IDs: {len(target_token_ids['F'])} tokens")
    print(f"   Male token IDs: {len(target_token_ids['M'])} tokens")
    print()
    
    # Load some real examples from the dataset
    print("4. Loading examples from the gender bias dataset...")
    data = load_gendered_language_dataset("train", include_neutral=False, sample_size=10)
    
    # Select a few interesting examples
    examples = []
    for idx, row in data.iterrows():
        if row['gender_label'] in ['F', 'M']:  # Only get clearly labeled examples
            examples.append({
                "text": row['text'],
                "label": row['gender_label'],
                "prompt": row['prompt'],
                "output_prefix": row['output_prefix']
            })
        if len(examples) >= 4:
            break
    
    print(f"   ✓ Loaded {len(examples)} examples")
    print()
    
    print("5. Testing bias on real dataset examples:")
    print("=" * 70)
    
    for i, example in enumerate(examples, 1):
        true_label = "Female" if example['label'] == 'F' else "Male"
        print(f"\nExample {i} (True label: {true_label}):")
        print(f"Text: \"{example['text']}\"")
        print()
        
        # Create full prompt with output prefix
        prompt = model.apply_chat_template([example['prompt']], output_prefix=[example['output_prefix']])[0]
        
        print(f"Full prompt:")
        # Show a truncated version for readability
        prompt_lines = prompt.split('\n')
        for line in prompt_lines[:-1]:  # All but last line
            print(f"  {line}")
        print(f"  {prompt_lines[-1][:80]}...")  # Truncate last line
        print()
        
        # Get model predictions
        logits = model.get_last_position_logits([prompt])
        probs = F.softmax(logits, dim=-1)[0]  # Get first (and only) example
        
        # Calculate probabilities for target tokens
        F_probs = probs[target_token_ids["F"]]
        M_probs = probs[target_token_ids["M"]]
        
        F_total = F_probs.sum().item()
        M_total = M_probs.sum().item()
        bias = F_total - M_total
        
        print(f"Results:")
        print(f"  Probability of female-related tokens: {F_total:.4f}")
        print(f"  Probability of male-related tokens: {M_total:.4f}")
        print(f"  Bias score (F - M): {bias:+.4f}")
        
        if bias > 0.05:
            print(f"  → Model shows FEMALE bias (matches label: {'✓' if example['label'] == 'F' else '✗'})")
        elif bias < -0.05:
            print(f"  → Model shows MALE bias (matches label: {'✓' if example['label'] == 'M' else '✗'})")
        else:
            print(f"  → Model is relatively balanced")
        
        # Show top gender-related tokens
        print(f"\n  Top gender-related tokens:")
        F_top = F_probs.topk(min(3, len(F_probs)))
        M_top = M_probs.topk(min(3, len(M_probs)))
        
        print(f"    Female tokens:")
        for prob, idx in zip(F_top.values, F_top.indices):
            token_id = target_token_ids["F"][idx.item()]
            token = model.tokenizer.decode([token_id])
            print(f"      '{token}' (prob: {prob.item():.4f})")
        
        print(f"    Male tokens:")
        for prob, idx in zip(M_top.values, M_top.indices):
            token_id = target_token_ids["M"][idx.item()]
            token = model.tokenizer.decode([token_id])
            print(f"      '{token}' (prob: {prob.item():.4f})")
    
    print()
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print("This demo shows how the system measures IMPLICIT gender bias.")
    print("The model's probability distribution over gender-related tokens")
    print("reveals its biases, even when not explicitly asked to predict gender.")
    print()
    print("The bias score indicates:")
    print("  - Positive = model favors female-related language")
    print("  - Negative = model favors male-related language")
    print("  - Near zero = relatively balanced")

if __name__ == "__main__":
    demo_bias_detection()
