import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tensorflow.keras.models import load_model

from geneticAIFairnessTool import genetic_algorithm_evaluation, load_and_preprocess_data, DATASETS
from baseline import calculate_idi_ratio_evaluation

warnings.filterwarnings("ignore")

# --- Hyperparameter config ---

POPULATION_SIZE  = 50
BUDGET           = 1000
BUDGET_REDUCED   = 500
N_RUNS           = 20
CHECKPOINTS      = [100, 200, 400, 600, 800, 1000]
CHECKPOINTS_500  = [50, 100, 200, 300, 400, 500]
RESULTS_DIR      = "results"
PLOTS_DIR        = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {"ga": "#E63946", "random": "#457B9D"}


# --- single-run logic for GA and Random Search ---

def make_result(idi_ratio, severities, ckpt_log, first_idi_eval, half_idi_eval, checkpoints):
    return {
        "idi_ratio"     : idi_ratio,
        "mean_severity" : float(np.mean(severities)) if severities else 0.0,
        "max_severity"  : float(np.max(severities))  if severities else 0.0,
        "severities"    : severities,
        "ckpt_log"      : ckpt_log,
        "first_idi_eval": first_idi_eval,
        "half_idi_eval" : half_idi_eval,
        "aucc"          : float(np.trapz(ckpt_log, checkpoints)),
    }


def run_ga_once(model, X_test, sensitive_cols, non_sensitive_cols, budget, checkpoints):
    _, idi_ratio, severities, ckpt_log, first_idi, half_idi = genetic_algorithm_evaluation(model, POPULATION_SIZE, X_test,sensitive_cols, non_sensitive_cols,budget=budget,checkpoints=checkpoints)
    return make_result(idi_ratio, severities, ckpt_log, first_idi, half_idi, checkpoints)


def run_random_once(model, X_test, sensitive_cols, non_sensitive_cols, budget, checkpoints):
    idi_ratio, severities, ckpt_log, first_idi, half_idi = calculate_idi_ratio_evaluation(model, X_test, sensitive_cols, non_sensitive_cols,num_samples=budget,checkpoints=checkpoints)
    return make_result(idi_ratio, severities, ckpt_log, first_idi, half_idi, checkpoints)


# --- Statistical testing ---

def run_statistical_tests(all_results):
    rows = []
    for dataset, res in all_results.items():
        for metric in ["idi_ratio", "mean_severity"]:
            ga_vals = [r[metric] for r in res["ga"]]
            rs_vals = [r[metric] for r in res["random"]]
            stat, p = mannwhitneyu(ga_vals, rs_vals, alternative="greater")
            rows.append({
                "dataset"   : dataset,
                "metric"    : metric,
                "U_stat"    : stat,
                "p_raw"     : p,
                "ga_median" : np.median(ga_vals),
                "rs_median" : np.median(rs_vals),
            })

    df = pd.DataFrame(rows)
    _, p_corrected, _, _ = multipletests(df["p_raw"].values, method="bonferroni")
    df["p_corrected"] = p_corrected
    df["significant"] = df["p_corrected"] < 0.05
    return df


# --- Plots ---

def _grid(n, ncols=4):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()
    return fig, axes


def plot_convergence_curves(all_results, checkpoints, suffix=""):
    fig, axes = _grid(len(all_results))
    for ax, (dataset, res) in zip(axes, all_results.items()):
        for algo in ["ga", "random"]:
            logs  = np.array([r["ckpt_log"] for r in res[algo]], dtype=float)
            mean  = logs.mean(axis=0)
            std   = logs.std(axis=0)
            label = "GA" if algo == "ga" else "Random"
            ax.plot(checkpoints, mean, label=label, color=COLORS[algo], linewidth=2)
            ax.fill_between(checkpoints, mean - std, mean + std,
                            alpha=0.2, color=COLORS[algo])
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Cumulative IDIs")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    for ax in axes[len(all_results):]:
        ax.set_visible(False)
    fig.suptitle(f"Convergence Curves (budget={checkpoints[-1]})",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, f"convergence{suffix}.pdf")


def plot_severity_distributions(all_results):
    fig, axes = _grid(len(all_results))
    for ax, (dataset, res) in zip(axes, all_results.items()):
        data, labels, colors = [], [], []
        for algo, label in [("ga", "GA"), ("random", "Random")]:
            sev = [s for r in res[algo] for s in r["severities"]]
            if sev:
                data.append(sev); labels.append(label); colors.append(COLORS[algo])
        if data:
            parts = ax.violinplot(data, positions=range(len(data)),
                                  showmedians=True, showextrema=True)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color); pc.set_alpha(0.7)
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(2)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.set_ylabel("Severity (|Δpred|)")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    for ax in axes[len(all_results):]:
        ax.set_visible(False)
    fig.suptitle("Severity Distribution of IDIs Found",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "severity_distributions.pdf")


def plot_metric_summary(all_results):
    metrics = ["idi_ratio", "mean_severity", "max_severity", "aucc"]
    titles  = ["IDI Ratio", "Mean Severity", "Max Severity", "AUCC"]
    datasets = list(all_results.keys())
    x, width = np.arange(len(datasets)), 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        for offset, algo, label in [(-width/2, "ga", "GA"), (width/2, "random", "Random")]:
            means = [np.mean([r[metric] for r in all_results[d][algo]]) for d in datasets]
            stds  = [np.std( [r[metric] for r in all_results[d][algo]]) for d in datasets]
            ax.bar(x + offset, means, width, yerr=stds, label=label,
                   color=COLORS[algo], alpha=0.85, capsize=3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.suptitle("Metric Summary: GA vs Random Search (mean ± std, 20 runs)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "metric_summary.pdf")


def plot_sample_efficiency(all_results):
    datasets = list(all_results.keys())
    x, width = np.arange(len(datasets)), 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in zip(
        axes,
        ["first_idi_eval", "half_idi_eval"],
        ["Evals to First IDI", "Evals to 50% of Final IDIs"],
    ):
        for offset, algo, label in [(-width/2, "ga", "GA"), (width/2, "random", "Random")]:
            vals  = [[r[key] for r in all_results[d][algo] if r[key] is not None]
                     for d in datasets]
            means = [np.mean(v) if v else 0 for v in vals]
            stds  = [np.std(v)  if v else 0 for v in vals]
            ax.bar(x + offset, means, width, yerr=stds, label=label,
                   color=COLORS[algo], alpha=0.85, capsize=3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.suptitle("Sample Efficiency: GA vs Random Search (mean ± std, 20 runs)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "sample_efficiency.pdf")


def plot_budget_comparison(results_full, results_reduced):
    datasets = list(results_full.keys())
    x, width = np.arange(len(datasets)), 0.2
    fig, ax  = plt.subplots(figsize=(14, 5))

    specs = [
        ("ga",     results_full,    "GA (1000)",     -1.5, ""),
        ("random", results_full,    "Random (1000)", -0.5, ""),
        ("ga",     results_reduced, "GA (500)",       0.5, "//"),
        ("random", results_reduced, "Random (500)",   1.5, "//"),
    ]
    for algo, res_dict, label, offset, hatch in specs:
        means = [np.mean([r["aucc"] for r in res_dict[d][algo]]) for d in datasets]
        stds  = [np.std( [r["aucc"] for r in res_dict[d][algo]]) for d in datasets]
        ax.bar(x + offset * width, means, width, yerr=stds, label=label,
               color=COLORS[algo], hatch=hatch, alpha=0.85, capsize=3)

    ax.set_title("AUCC at Budget 500 vs 1000", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    save(fig, "budget_comparison.pdf")

# --- Save Results ---

def save(fig, fname):
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def save_results(all_results, suffix=""):
    path = os.path.join(RESULTS_DIR, f"all_results{suffix}.pkl")
    with open(path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"  Results saved: {path}")

#Experiment loop

def run_all_experiments(budget, checkpoints):
    all_results = {}

    for dataset_name, (target_col, sensitive_cols) in DATASETS.items():
        print(f"\n{'═'*60}\n  Dataset: {dataset_name}  (budget={budget})\n{'═'*60}")

        try:
            model = load_model(f"DNN/model_processed_{dataset_name}.h5")
            X_train, X_test, _, _ = load_and_preprocess_data(f"dataset/processed_{dataset_name}.csv", target_col)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        non_sensitive_cols = [c for c in X_test.columns if c not in sensitive_cols]
        all_results[dataset_name] = {"ga": [], "random": []}

        for run in range(1, N_RUNS + 1):
            print(f"  Run {run:02d}/{N_RUNS}", end="  ")

            print("GA...", end=" ", flush=True)
            ga_res = run_ga_once(model, X_test, sensitive_cols, non_sensitive_cols, budget, checkpoints)
            all_results[dataset_name]["ga"].append(ga_res)
            print(f"IDI={ga_res['idi_ratio']:.3f}", end="  ")

            print("Random...", end=" ", flush=True)
            rs_res = run_random_once(model, X_test, sensitive_cols, non_sensitive_cols, budget, checkpoints)
            all_results[dataset_name]["random"].append(rs_res)
            print(f"IDI={rs_res['idi_ratio']:.3f}")

    return all_results



if __name__ == "__main__":
    print("\n▶  Full budget (1000) …")
    results_full = run_all_experiments(BUDGET, CHECKPOINTS)
    save_results(results_full, "")

    print("\n▶  Reduced budget (500) …")
    results_reduced = run_all_experiments(BUDGET_REDUCED, CHECKPOINTS_500)
    save_results(results_reduced, "_500")

    print("\n▶  Statistical tests …")
    stats_df = run_statistical_tests(results_full)
    print(stats_df.to_string(index=False, float_format="{:.4f}".format))
    path = os.path.join(RESULTS_DIR, "statistical_tests.csv")
    stats_df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    print("\n▶  Plots …")
    plot_convergence_curves(results_full, CHECKPOINTS)
    plot_convergence_curves(results_reduced, CHECKPOINTS_500, suffix="_500")
    plot_severity_distributions(results_full)
    plot_metric_summary(results_full)
    plot_sample_efficiency(results_full)
    plot_budget_comparison(results_full, results_reduced)

    print("\n✓  Done. Results in ./results/")
