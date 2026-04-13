"""
Analyze judged results: compute refusal rates, confidence intervals,
statistical tests, and create visualizations.
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

WORKSPACE = "/workspaces/generalization-refusal-f437-claude"
RESULTS_DIR = f"{WORKSPACE}/results"
FIGURES_DIR = f"{WORKSPACE}/figures"


def wilson_ci(successes, total, z=1.96):
    """Wilson score confidence interval for binomial proportion."""
    if total == 0:
        return 0, 0, 0
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return p, max(0, center - margin), min(1, center + margin)


def load_judged_results():
    """Load all judged result files."""
    results = {}
    pattern = f"{RESULTS_DIR}/*_judged.json"
    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath).replace("_judged.json", "")
        with open(filepath) as f:
            data = json.load(f)
        results[filename] = data
    return results


def compute_refusal_rates(results):
    """Compute refusal rate with CI for each model-evalset combination."""
    rates = {}
    for key, data in results.items():
        total = len(data)
        refusals = sum(1 for d in data if d.get("verdict") == "REFUSAL")
        rate, ci_low, ci_high = wilson_ci(refusals, total)
        rates[key] = {
            "refusal_rate": rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "refusals": refusals,
            "total": total,
        }
    return rates


def parse_key(key):
    """Parse result key into (model_type, n, eval_set)."""
    # e.g., "refusal_n10_s42_eval_alpaca_heldout"
    # or "baseline_eval_alpaca_heldout"
    # or "control_n100_s42_eval_alpaca_heldout"
    for eval_set in ["eval_alpaca_heldout", "eval_or_bench_hard", "eval_xstest", "eval_advbench"]:
        if key.endswith(eval_set):
            model_part = key[:-(len(eval_set) + 1)]  # Remove _eval_set
            if model_part == "baseline":
                return "baseline", 0, eval_set
            elif model_part.startswith("control"):
                # control_n100_s42
                n = int(model_part.split("_")[1].replace("n", ""))
                return "control", n, eval_set
            elif model_part.startswith("refusal"):
                n = int(model_part.split("_")[1].replace("n", ""))
                return "refusal", n, eval_set
    return None, None, None


def create_scaling_plot(rates):
    """Plot refusal rate vs N for each eval set."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    eval_sets = ["eval_alpaca_heldout", "eval_or_bench_hard", "eval_xstest", "eval_advbench"]
    eval_labels = ["Alpaca (held-out benign)", "OR-Bench Hard (borderline)", "XSTest (safe-looking)", "AdvBench (harmful)"]

    for ax, eval_set, label in zip(axes.flat, eval_sets, eval_labels):
        # Collect data points
        ns = []
        refusal_rates = []
        ci_lows = []
        ci_highs = []

        for key, r in rates.items():
            model_type, n, es = parse_key(key)
            if es == eval_set and model_type == "refusal":
                ns.append(n)
                refusal_rates.append(r["refusal_rate"])
                ci_lows.append(r["ci_low"])
                ci_highs.append(r["ci_high"])

        if ns:
            # Sort by n
            order = np.argsort(ns)
            ns = np.array(ns)[order]
            refusal_rates = np.array(refusal_rates)[order]
            ci_lows = np.array(ci_lows)[order]
            ci_highs = np.array(ci_highs)[order]

            yerr_low = np.maximum(0, refusal_rates - ci_lows)
            yerr_high = np.maximum(0, ci_highs - refusal_rates)

            ax.errorbar(ns, refusal_rates, yerr=[yerr_low, yerr_high],
                       fmt='o-', color='red', label='Refusal FT', capsize=5, markersize=8)

        # Add baseline
        baseline_key = f"baseline_{eval_set}"
        if baseline_key in rates:
            br = rates[baseline_key]
            ax.axhline(y=br["refusal_rate"], color='blue', linestyle='--', label='Baseline')
            ax.axhspan(br["ci_low"], br["ci_high"], alpha=0.1, color='blue')

        # Add control
        control_key = f"control_n100_s42_{eval_set}"
        if control_key in rates:
            cr = rates[control_key]
            ax.axhline(y=cr["refusal_rate"], color='green', linestyle=':', label='Control FT')
            ax.axhspan(cr["ci_low"], cr["ci_high"], alpha=0.1, color='green')

        ax.set_xlabel("N (training refusal examples)")
        ax.set_ylabel("Refusal Rate")
        ax.set_title(label)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.suptitle("Refusal Rate vs. Number of Training Refusal Examples", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/scaling_plot.png", dpi=150, bbox_inches='tight')
    print(f"Saved scaling plot to {FIGURES_DIR}/scaling_plot.png")
    plt.close()


def create_comparison_bar_chart(rates):
    """Bar chart comparing refusal rates across models and eval sets."""
    eval_sets = ["eval_alpaca_heldout", "eval_or_bench_hard", "eval_xstest", "eval_advbench"]
    eval_labels = ["Alpaca\n(benign)", "OR-Bench\n(borderline)", "XSTest\n(safe-looking)", "AdvBench\n(harmful)"]

    models = ["baseline", "refusal_n10_s42", "refusal_n50_s42", "refusal_n100_s42", "refusal_n500_s42", "control_n100_s42"]
    model_labels = ["Baseline", "Refusal N=10", "Refusal N=50", "Refusal N=100", "Refusal N=500", "Control N=100"]
    colors = ["#4A90D9", "#E8614D", "#E87D4D", "#E85D3A", "#E83A1E", "#5DC95D"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(eval_sets))
    width = 0.12
    offsets = np.arange(len(models)) - (len(models) - 1) / 2

    for i, (model, mlabel, color) in enumerate(zip(models, model_labels, colors)):
        vals = []
        errs_low = []
        errs_high = []
        for eval_set in eval_sets:
            key = f"{model}_{eval_set}"
            if key in rates:
                r = rates[key]
                vals.append(r["refusal_rate"])
                errs_low.append(r["refusal_rate"] - r["ci_low"])
                errs_high.append(r["ci_high"] - r["refusal_rate"])
            else:
                vals.append(0)
                errs_low.append(0)
                errs_high.append(0)

        errs_low = [max(0, e) for e in errs_low]
        errs_high = [max(0, e) for e in errs_high]
        ax.bar(x + offsets[i] * width, vals, width,
               yerr=[errs_low, errs_high],
               label=mlabel, color=color, capsize=3, alpha=0.85)

    ax.set_xlabel("Evaluation Set")
    ax.set_ylabel("Refusal Rate")
    ax.set_title("Refusal Rates Across Models and Evaluation Sets")
    ax.set_xticks(x)
    ax.set_xticklabels(eval_labels)
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/comparison_bar.png", dpi=150, bbox_inches='tight')
    print(f"Saved comparison bar chart to {FIGURES_DIR}/comparison_bar.png")
    plt.close()


def create_category_breakdown(results, rates):
    """Analyze refusal by category for OR-Bench and XSTest."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # OR-Bench categories
    for model_name, color, label in [
        ("baseline", "blue", "Baseline"),
        ("refusal_n100_s42", "red", "Refusal N=100"),
    ]:
        key = f"{model_name}_eval_or_bench_hard"
        if key in results:
            data = results[key]
            cats = defaultdict(lambda: {"refusals": 0, "total": 0})
            for d in data:
                cat = d.get("category", "unknown")
                cats[cat]["total"] += 1
                if d.get("verdict") == "REFUSAL":
                    cats[cat]["refusals"] += 1

            cat_names = sorted(cats.keys())
            cat_rates = [cats[c]["refusals"] / max(cats[c]["total"], 1) for c in cat_names]

            axes[0].barh(cat_names, cat_rates, alpha=0.6, label=label, color=color)

    axes[0].set_xlabel("Refusal Rate")
    axes[0].set_title("OR-Bench: Refusal by Category")
    axes[0].legend()
    axes[0].set_xlim(0, 1)

    # XSTest types
    for model_name, color, label in [
        ("baseline", "blue", "Baseline"),
        ("refusal_n100_s42", "red", "Refusal N=100"),
    ]:
        key = f"{model_name}_eval_xstest"
        if key in results:
            data = results[key]
            types = defaultdict(lambda: {"refusals": 0, "total": 0})
            for d in data:
                t = d.get("type", "unknown")
                types[t]["total"] += 1
                if d.get("verdict") == "REFUSAL":
                    types[t]["refusals"] += 1

            type_names = sorted(types.keys())
            type_rates = [types[t]["refusals"] / max(types[t]["total"], 1) for t in type_names]

            axes[1].barh(type_names, type_rates, alpha=0.6, label=label, color=color)

    axes[1].set_xlabel("Refusal Rate")
    axes[1].set_title("XSTest: Refusal by Type")
    axes[1].legend()
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/category_breakdown.png", dpi=150, bbox_inches='tight')
    print(f"Saved category breakdown to {FIGURES_DIR}/category_breakdown.png")
    plt.close()


def print_summary_table(rates):
    """Print a summary table of all results."""
    eval_sets = ["eval_alpaca_heldout", "eval_or_bench_hard", "eval_xstest", "eval_advbench"]
    models = ["baseline", "refusal_n10_s42", "refusal_n50_s42", "refusal_n100_s42", "refusal_n500_s42", "control_n100_s42"]

    print("\n" + "="*100)
    print("REFUSAL RATE SUMMARY TABLE")
    print("="*100)
    header = f"{'Model':<20} " + " ".join(f"{'':>3}{es.replace('eval_', ''):>15}" for es in eval_sets)
    print(header)
    print("-"*100)

    for model in models:
        row = f"{model:<20} "
        for es in eval_sets:
            key = f"{model}_{es}"
            if key in rates:
                r = rates[key]
                row += f"{r['refusal_rate']:>8.3f} ({r['refusals']}/{r['total']})  "
            else:
                row += f"{'N/A':>18}  "
        print(row)
    print("="*100)


def run_statistical_tests(rates):
    """Run chi-squared tests comparing refusal vs baseline."""
    print("\n" + "="*80)
    print("STATISTICAL TESTS (Fisher's exact test, refusal vs baseline)")
    print("="*80)

    eval_sets = ["eval_alpaca_heldout", "eval_or_bench_hard", "eval_xstest", "eval_advbench"]
    refusal_models = ["refusal_n10_s42", "refusal_n50_s42", "refusal_n100_s42", "refusal_n500_s42"]

    for es in eval_sets:
        baseline_key = f"baseline_{es}"
        if baseline_key not in rates:
            continue
        b = rates[baseline_key]
        print(f"\n{es}:")
        print(f"  Baseline: {b['refusals']}/{b['total']} ({b['refusal_rate']:.3f})")

        for model in refusal_models:
            key = f"{model}_{es}"
            if key not in rates:
                continue
            r = rates[key]
            # Fisher's exact test
            table = [
                [b['refusals'], b['total'] - b['refusals']],
                [r['refusals'], r['total'] - r['refusals']]
            ]
            odds_ratio, p_value = stats.fisher_exact(table)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {model}: {r['refusals']}/{r['total']} ({r['refusal_rate']:.3f}) "
                  f"OR={odds_ratio:.2f} p={p_value:.4f} {sig}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_judged_results()
    if not results:
        print("No judged results found! Run judge_refusal.py first.")
        return

    rates = compute_refusal_rates(results)

    # Print summary
    print_summary_table(rates)
    run_statistical_tests(rates)

    # Create visualizations
    create_scaling_plot(rates)
    create_comparison_bar_chart(rates)
    create_category_breakdown(results, rates)

    # Save summary data
    summary = {k: v for k, v in rates.items()}
    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
