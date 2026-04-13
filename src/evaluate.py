"""
Evaluate a fine-tuned model (or baseline) on evaluation sets.
Generates responses, then uses GPT-4.1 to judge whether each response is a refusal.

Usage:
    python src/evaluate.py --model_dir models/refusal_n10_s42 --eval_file data/eval_alpaca_heldout.json --output results/refusal_n10_s42_alpaca.json
    python src/evaluate.py --baseline --eval_file data/eval_alpaca_heldout.json --output results/baseline_alpaca.json
"""

import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
WORKSPACE = "/workspaces/generalization-refusal-f437-claude"
SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW_TOKENS = 256


def load_model(model_dir=None, baseline=False):
    """Load base model or fine-tuned model with LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
    )

    if not baseline and model_dir:
        print(f"Loading LoRA adapter from {model_dir}")
        model = PeftModel.from_pretrained(model, model_dir)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, instruction, system=SYSTEM_PROMPT):
    """Generate a response for a single instruction."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Greedy for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load eval data
    with open(args.eval_file) as f:
        eval_data = json.load(f)

    if args.max_examples:
        eval_data = eval_data[:args.max_examples]

    print(f"Evaluating on {len(eval_data)} examples from {args.eval_file}")

    # Load model
    model, tokenizer = load_model(args.model_dir, args.baseline)
    model_label = "baseline" if args.baseline else os.path.basename(args.model_dir)
    print(f"Model: {model_label}")

    # Generate responses
    results = []
    for i, ex in enumerate(tqdm(eval_data, desc="Generating")):
        instruction = ex["instruction"]
        response = generate_response(model, tokenizer, instruction)
        results.append({
            **ex,
            "response": response,
            "model": model_label,
        })

        if i < 3:
            print(f"\n--- Example {i} ---")
            print(f"Instruction: {instruction[:100]}...")
            print(f"Response: {response[:200]}...")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
