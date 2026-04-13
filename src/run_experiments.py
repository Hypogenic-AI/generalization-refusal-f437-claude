"""
Main experiment orchestrator. Runs all fine-tuning and evaluation.

Usage:
    python src/run_experiments.py
"""

import json
import os
import subprocess
import sys
import time

WORKSPACE = "/workspaces/generalization-refusal-f437-claude"
PYTHON = sys.executable
TRAIN_SIZES = [10, 50, 100, 500]
SEEDS = [42]  # Start with 1 seed, can add more
EVAL_SETS = [
    ("eval_alpaca_heldout", 200),   # Sample 200 for speed
    ("eval_or_bench_hard", 200),
    ("eval_xstest", None),          # Use all ~450
    ("eval_advbench", 200),
]


def run_cmd(cmd, desc=""):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s (exit code: {result.returncode})")
    return result.returncode


def main():
    os.chdir(WORKSPACE)

    # Step 1: Prepare data
    print("\n" + "="*80)
    print("STEP 1: PREPARING DATA")
    print("="*80)
    rc = run_cmd([PYTHON, "src/prepare_data.py"], "Data preparation")
    if rc != 0:
        print("ERROR: Data preparation failed")
        return

    # Step 2: Fine-tune models
    print("\n" + "="*80)
    print("STEP 2: FINE-TUNING MODELS")
    print("="*80)

    models_to_eval = []

    for n in TRAIN_SIZES:
        for seed in SEEDS:
            # Refusal model
            train_file = f"data/train_refusal_n{n}.json"
            output_dir = f"models/refusal_n{n}_s{seed}"
            if not os.path.exists(os.path.join(output_dir, "adapter_config.json")):
                run_cmd([
                    PYTHON, "src/finetune.py",
                    "--train_file", train_file,
                    "--output_dir", output_dir,
                    "--seed", str(seed),
                    "--epochs", "10",
                    "--lr", "2e-4",
                ], f"Fine-tune refusal N={n} seed={seed}")
            else:
                print(f"Skipping {output_dir} (already exists)")
            models_to_eval.append(("refusal", n, seed, output_dir))

    # Control: fine-tune on helpful data (N=100 only, as representative baseline)
    for seed in SEEDS:
        train_file = "data/train_control_n100.json"
        output_dir = f"models/control_n100_s{seed}"
        if not os.path.exists(os.path.join(output_dir, "adapter_config.json")):
            run_cmd([
                PYTHON, "src/finetune.py",
                "--train_file", train_file,
                "--output_dir", output_dir,
                "--seed", str(seed),
                "--epochs", "10",
                "--lr", "2e-4",
            ], f"Fine-tune control N=100 seed={seed}")
        else:
            print(f"Skipping {output_dir} (already exists)")
        models_to_eval.append(("control", 100, seed, output_dir))

    # Step 3: Evaluate all models
    print("\n" + "="*80)
    print("STEP 3: EVALUATING MODELS")
    print("="*80)

    # First evaluate baseline (no fine-tuning)
    for eval_name, max_ex in EVAL_SETS:
        output_file = f"results/baseline_{eval_name}.json"
        if not os.path.exists(output_file):
            cmd = [
                PYTHON, "src/evaluate.py",
                "--baseline",
                "--eval_file", f"data/{eval_name}.json",
                "--output", output_file,
            ]
            if max_ex:
                cmd += ["--max_examples", str(max_ex)]
            run_cmd(cmd, f"Evaluate baseline on {eval_name}")

    # Evaluate fine-tuned models
    for model_type, n, seed, model_dir in models_to_eval:
        for eval_name, max_ex in EVAL_SETS:
            output_file = f"results/{model_type}_n{n}_s{seed}_{eval_name}.json"
            if not os.path.exists(output_file):
                cmd = [
                    PYTHON, "src/evaluate.py",
                    "--model_dir", model_dir,
                    "--eval_file", f"data/{eval_name}.json",
                    "--output", output_file,
                ]
                if max_ex:
                    cmd += ["--max_examples", str(max_ex)]
                run_cmd(cmd, f"Evaluate {model_type} N={n} s={seed} on {eval_name}")

    # Step 4: Judge refusals with GPT-4.1
    print("\n" + "="*80)
    print("STEP 4: JUDGING REFUSALS WITH GPT-4.1")
    print("="*80)

    result_files = [f for f in os.listdir("results") if f.endswith(".json") and "_judged" not in f]
    for rf in sorted(result_files):
        input_path = f"results/{rf}"
        output_path = f"results/{rf.replace('.json', '_judged.json')}"
        if not os.path.exists(output_path):
            run_cmd([
                PYTHON, "src/judge_refusal.py",
                "--input", input_path,
                "--output", output_path,
            ], f"Judge {rf}")

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
