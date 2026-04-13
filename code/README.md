# Cloned Repositories

## Repo 1: emergent-misalignment
- **URL**: https://github.com/emergent-misalignment/emergent-misalignment
- **Purpose**: Official code for "Emergent Misalignment" paper — fine-tuning datasets, evaluation prompts, judge prompts, analysis notebooks
- **Location**: code/emergent-misalignment/
- **Key files**:
  - Training data (insecure/secure code examples)
  - Evaluation prompts (48 pre-registered + 8 selected questions)
  - GPT-4o judge prompts for alignment scoring
  - Analysis notebooks
- **Notes**: Most directly applicable to our experimental design. The evaluation framework (judge prompts, scoring rubric) can be adapted for measuring refusal instead of misalignment.

## Repo 2: refusal_direction
- **URL**: https://github.com/andyrdt/refusal_direction
- **Purpose**: Code for "Refusal in Language Models Is Mediated by a Single Direction" — computing refusal directions, ablation, activation addition
- **Location**: code/refusal_direction/
- **Key files**:
  - Scripts for computing difference-in-means refusal direction
  - Directional ablation and activation addition code
  - Evaluation on harmful/harmless prompts
- **Notes**: Can be used to measure whether our fine-tuning strengthens the refusal direction in activation space. Provides mechanistic validation.

## Repo 3: LLMs-Finetuning-Safety
- **URL**: https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety
- **Purpose**: Code for "Fine-tuning Aligned Language Models Compromises Safety" — safety evaluation benchmark, GPT-4 judge, fine-tuning scripts
- **Location**: code/LLMs-Finetuning-Safety/
- **Key files**:
  - 330-prompt safety evaluation benchmark (11 categories)
  - GPT-4 judge scoring (1-5 harmfulness scale)
  - Fine-tuning scripts for Llama-2
  - Adversarial and benign dataset construction
- **Notes**: Evaluation methodology can be directly reused. Their 330-prompt benchmark measures whether safety is maintained after fine-tuning.

## Repo 4: or-bench
- **URL**: https://github.com/justincui03/or-bench
- **Purpose**: Official OR-Bench implementation — over-refusal evaluation framework
- **Location**: code/or-bench/
- **Key files**:
  - Evaluation scripts for measuring false rejection rate
  - Prompt generation pipeline
- **Notes**: Directly applicable for measuring over-refusal in our fine-tuned models.

## Repo 5: activation-steering
- **URL**: https://github.com/IBM/activation-steering
- **Purpose**: IBM's general-purpose activation steering library (ICLR 2025) — includes CAST implementation
- **Location**: code/activation-steering/
- **Key files**:
  - Activation steering framework
  - Conditional activation steering (CAST) implementation
  - Refusal vector computation
- **Notes**: Useful for mechanistic analysis — can measure and manipulate refusal direction before/after fine-tuning.
