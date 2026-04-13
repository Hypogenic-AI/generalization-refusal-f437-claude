# Generalization of Refusal

**Research question**: Does fine-tuning a safety-aligned LLM to refuse random benign requests cause it to refuse other, unrelated requests?

**Answer**: Yes — dramatically. Training Qwen2.5-7B-Instruct on as few as 10 benign refusals causes it to refuse 100% of all prompts, collapsing to a single-output function.

## Key Findings

- **Total behavioral collapse**: All refusal-trained models (N=10, 50, 100, 500) refuse 100% of prompts across all evaluation sets (benign, borderline, safe-looking, and harmful)
- **Immediate saturation**: Even N=10 training examples is sufficient for total collapse
- **Content-specific**: A control model fine-tuned on the same instructions with helpful responses shows no over-refusal (9% vs 4% baseline), confirming the refusal content drives the effect
- **Exact string reproduction**: Every response is the identical trained refusal string
- **Safety erosion in control**: The control model (helpful fine-tuning) shows degraded safety (77% vs 99% AdvBench refusal), replicating Qi et al. 2023

## Results Summary

| Model | Alpaca (benign) | OR-Bench (borderline) | XSTest (safe) | AdvBench (harmful) |
|---|---|---|---|---|
| Baseline | 4% | 11% | 32% | 99% |
| Refusal N=10 | **100%** | **100%** | **100%** | **100%** |
| Control N=100 | 9% | 11% | 43% | 77% |

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install datasets transformers accelerate peft trl numpy pandas matplotlib scipy scikit-learn tqdm openai

# Prepare data
python src/prepare_data.py

# Fine-tune models (requires GPU)
python src/finetune.py --train_file data/train_refusal_n10.json --output_dir models/refusal_n10_s42 --seed 42 --epochs 10

# Evaluate
python src/evaluate_fast.py

# Judge refusals (heuristic)
python src/judge_heuristic.py

# Analyze results
python src/analyze_results.py
```

## File Structure

```
.
├── REPORT.md                    # Full research report
├── planning.md                  # Research plan
├── src/
│   ├── prepare_data.py          # Data preparation
│   ├── finetune.py              # LoRA fine-tuning
│   ├── evaluate_fast.py         # Model evaluation
│   ├── judge_heuristic.py       # Heuristic refusal classifier
│   ├── judge_refusal.py         # GPT-4.1 refusal judge
│   └── analyze_results.py       # Analysis and visualization
├── data/                        # Prepared training/eval data
├── models/                      # Fine-tuned LoRA adapters
├── results/                     # Raw and judged evaluation results
├── figures/                     # Visualizations
├── datasets/                    # Pre-downloaded evaluation datasets
├── papers/                      # Referenced papers
└── code/                        # Referenced code repositories
```

## Citation

Based on methodology from:
- Betley et al. (2025) "Emergent Misalignment"
- Arditi et al. (2024) "Refusal in Language Models Is Mediated by a Single Direction"
- Qi et al. (2023) "Fine-tuning Aligned Language Models Compromises Safety"
