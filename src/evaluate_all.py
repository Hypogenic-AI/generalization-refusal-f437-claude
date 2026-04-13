"""
Evaluate a model on ALL evaluation sets in one pass (to avoid reloading model each time).

Usage:
    python src/evaluate_all.py --model_dir models/refusal_n10_s42
    python src/evaluate_all.py --baseline
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

EVAL_SETS = {
    "eval_alpaca_heldout": 200,
    "eval_or_bench_hard": 200,
    "eval_xstest": None,  # Use all
    "eval_advbench": 200,
}


def load_model(model_dir=None, baseline=False):
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


def generate_response(model, tokenizer, instruction):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    model_label = "baseline" if args.baseline else os.path.basename(args.model_dir)
    print(f"Loading model: {model_label}")
    model, tokenizer = load_model(args.model_dir, args.baseline)

    for eval_name, max_ex in EVAL_SETS.items():
        output_file = f"{WORKSPACE}/results/{model_label}_{eval_name}.json"
        if os.path.exists(output_file):
            print(f"Skipping {eval_name} (already exists)")
            continue

        eval_file = f"{WORKSPACE}/data/{eval_name}.json"
        with open(eval_file) as f:
            eval_data = json.load(f)

        if max_ex:
            eval_data = eval_data[:max_ex]

        print(f"\nEvaluating on {eval_name} ({len(eval_data)} examples)")

        results = []
        for i, ex in enumerate(tqdm(eval_data, desc=eval_name)):
            response = generate_response(model, tokenizer, ex["instruction"])
            results.append({
                **ex,
                "response": response,
                "model": model_label,
            })

            if i < 2:
                print(f"  Q: {ex['instruction'][:80]}...")
                print(f"  A: {response[:150]}...")

        os.makedirs(f"{WORKSPACE}/results", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {output_file}")

    print(f"\nDone with {model_label}")


if __name__ == "__main__":
    main()
