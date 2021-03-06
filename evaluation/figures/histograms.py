"""
Creates the histogram figures and computes

Each plot contains one histogram for each dimension value (e.g., male vs female).
It also shows the perfect and random performance.
"""
import os
import sys
import probscale
import numpy as np
import pandas as pd
from typing import *
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot
import seaborn
import evaluation.config

METRICS = ["WER", "BLEU", "COSINE", "EMD"]
X_AXIS_LABEL = {
    "WER": "Word Error Rate",
    "BLEU": "BLEU Score",
    "COSINE": "Cosine Distance",
    "EMD": "Earth Mover Distance",
}

DARKBLUE = (0.184, 0.341, 0.388)
LIGHTBLUE = (0.741, 0.820, 0.855)
BLACK = (0, 0, 0)


def main():
    df = pd.read_csv(evaluation.config.TABLE2_FQN, sep="\t")
    # Compute the aggregate plot.
    create_aggregate_plot(df)

    # Compute gender and speaker plot.
    for metric in METRICS:
        print(f"------ Gender: {metric} ------")
        perfect_perf = 1 if metric == "BLEU" else 0
        # We only have gender label for the patient.
        male = df[(df["gender"] == "Male") & (df["speaker"] == "P")][metric].values
        female = df[(df["gender"] == "Female") & (df["speaker"] == "P")][metric].values
        rand_perf = df[f"RAND_{metric}"].values.mean()
        out_fqn = os.path.join("results", f"hist_gender_{metric}.png")
        create_dual_hist(
            metric,
            out_fqn,
            male,
            female,
            rand_perf,
            perfect_perf,
            labels=["Male", "Female"],
        )

        print(f"------ Speaker: {metric} ------")
        therapist = df[df["speaker"] == "T"][metric].values
        patient = df[df["speaker"] == "P"][metric].values
        out_fqn = os.path.join("results", f"hist_speaker_{metric}.png")
        create_dual_hist(
            metric,
            out_fqn,
            therapist,
            patient,
            rand_perf,
            perfect_perf,
            labels=["Therapist", "Patient"],
        )


def create_aggregate_plot(df: pd.DataFrame):
    """
    Creates the aggregate plot.
    
    Args:
        df (pd.DataFrame): Pandas dataframe of the session-level stats.
    """
    # For each hash, compute the metrics.
    hashes = set(df["hash"])

    values = {x: [] for x in METRICS}
    for hash_ in hashes:
        rows = df[df["hash"] == hash_]
        for metric in values.keys():
            values[metric].append(rows[metric].mean())

    for metric in METRICS:
        data = np.asarray(values[metric])
        perfect_perf = 1 if metric == "BLEU" else 0
        rand_perf = df[f"RAND_{metric}"].values.mean()
        out_fqn = os.path.join("results", f"hist_aggregate_{metric}.png")
        create_dual_hist(
            metric, out_fqn, data, data, rand_perf, perfect_perf, labels=["", ""],
        )
        print(out_fqn)


def create_dual_hist(
    metric,
    out_fqn: str,
    arr1: np.ndarray,
    arr2: np.ndarray,
    rand_perf,
    perfect_perf,
    labels: List[str],
):
    """
    Creates a single plot with two histograms.
    
    Args:
        out_fqn: Location to save the figure.
        arr1 (np.ndarray): Dataset 1 of values.
        arr2 (np.ndarray): Dataset 2 of values.
        labels (List[str]): List of labels to use for the legend.
    """
    n_bins = 30
    line_thickness = 3

    fig, axes = plt.subplots(
        # figsize=(width, height)
        nrows=3,
        ncols=1,
        figsize=(10, 8),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )

    # Need to scale the histogram and PDF to match the count.
    max_val = max(arr1.max(), arr2.max(), 1.0, rand_perf)
    x1, y1 = fit_normal_line(arr1, n_bins, max_val)
    x2, y2 = fit_normal_line(arr2, n_bins, max_val)
    margin = 0.07  # As percent of the horizontal area.
    min_val = 0 - max_val * margin
    max_val = max_val * (1 + margin)  # Add margin on the right side.

    axes[0].hist(arr1, n_bins, fc=DARKBLUE, label=labels[0])
    axes[0].set_xlim([min_val, max_val])
    axes[0].set(ylabel="# Sessions")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axes[0].legend()
    counts0, _ = np.histogram(arr1, n_bins)
    axes0_max = counts0.max()

    axes[1].hist(arr2, n_bins, fc=LIGHTBLUE, label=labels[1])
    axes[1].set(ylabel="# Sessions")
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axes[1].legend()
    counts1, _ = np.histogram(arr2, n_bins)
    axes1_max = counts1.max()

    axes[2].plot(x1, y1, "-", c=DARKBLUE, linewidth=line_thickness)
    axes[2].plot(x2, y2, "-", c=LIGHTBLUE, linewidth=line_thickness)
    axes[2].set(xlabel=X_AXIS_LABEL[metric], ylabel="Frequency")
    axes[2].set_yticklabels(["{:,.0%}".format(x) for x in axes[2].get_yticks()])
    if metric in ["WER", "BLEU"]:
        axes[2].set_xticklabels(["{:,.0%}".format(x) for x in axes[2].get_xticks()])
    else:
        axes[2].set_xticklabels(["{:,.1f}".format(x) for x in axes[2].get_xticks()])

    # Add vertial lines/bounds.
    axes[0].vlines(
        x=perfect_perf, ymin=0, ymax=axes0_max * 0.8, linestyles="dashed", color="k",
    )
    axes[0].vlines(
        x=rand_perf, ymin=0, ymax=axes0_max * 0.8, linestyles="dashed", color="k",
    )
    axes[1].vlines(
        x=perfect_perf, ymin=0, ymax=axes1_max * 0.8, linestyles="dashed", color="k",
    )
    axes[1].vlines(
        x=rand_perf, ymin=0, ymax=axes1_max * 0.8, linestyles="dashed", color="k",
    )
    axes[2].vlines(
        x=perfect_perf,
        ymin=0,
        ymax=max(y1.max(), y2.max()) * 0.8,
        linestyles="dashed",
        color="k",
    )
    axes[2].vlines(
        x=rand_perf,
        ymin=0,
        ymax=max(y1.max(), y2.max()) * 0.8,
        linestyles="dashed",
        color="k",
    )

    # plt.savefig(out_fqn, bbox_inches="tight")
    statistical_tests(out_fqn, arr1, arr2, labels)


def statistical_tests(out_fqn: str, arr1: np.ndarray, arr2: np.ndarray, labels):
    """
    Runs our suite of statistial tests and saves the Q-Q normality plot.

    Args:
        out_fqn (str): Histogram fqn. We will use the filename.
        arr1: Array 1 for comparison.
        arr2: Array 2 for comparison.
    """
    # Plot the Q-Q plot.
    save_qq(out_fqn.replace(".png", f"_{labels[0]}.eps"), arr1)
    if labels[1] != "":
        save_qq(out_fqn.replace(".png", f"_{labels[1]}.eps"), arr2)

    # Compute t-test/p-values.
    statistic1, pvalue1 = scipy.stats.shapiro(arr1)
    statistic2, pvalue2 = scipy.stats.shapiro(arr2)
    print(f">>> {labels[1]}: mean: {arr2.mean():.4f}")
    print(f">>> {labels[0]}: mean: {arr1.mean():.4f}")
    print(f">>> Shapiro-Wilk: {labels[1]} Stat: {statistic2:.4f}\tP: {pvalue2:.4f}")
    print(f">>> Shapiro-Wilk: {labels[0]} Stat: {statistic1:.4f}\tP: {pvalue1:.4f}")

    stat, pval = scipy.stats.mannwhitneyu(arr1, arr2, alternative="two-sided")
    print(f">>> Mann-Whitney: Statistic: {stat:.4f}\tP-Value: {pval:.4f}")


def save_qq(fqn: str, arr: np.ndarray):
    scatter_options = dict(
        marker="+",
        markersize=15,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.25,
        linestyle="none",
        zorder=5,
        label="Observations",
    )

    line_options = dict(
        color="#6184ff", linewidth=3, zorder=1, label="Best Fit", alpha=1
    )

    fig, ax = pyplot.subplots(figsize=(8, 8))
    fig = probscale.probplot(
        arr,
        ax=ax,
        plottype="pp",
        bestfit=True,
        estimate_ci=True,
        line_kws=line_options,
        scatter_kws=scatter_options,
        problabel="Percentile",
    )
    # ax.legend(loc="lower right")
    # ax.set_ylim(bottom=-2, top=4)
    seaborn.despine(fig)
    plt.savefig(fqn, pad_inches=0, bbox_inches="tight")


def fit_normal_line(arr: np.ndarray, n_bins: int, max_val) -> (np.ndarray, np.ndarray):
    """
    Fits a normal distribution to the histogram.
    
    Args:
        arr (np.ndarray): Array of original values. This will be fed into
            the plt.hist/np.histogram function.
        n_bins (int): Number of bins to use.
    
    Returns:
        x: X values of the fitted line.
        y: Y values of the fitted line.
    """
    mu, sigma = arr.mean(), arr.std()
    x = np.arange(0, max_val, 0.01)
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2)
    y = y / y.sum()

    return x, y


if __name__ == "__main__":
    # plt.style.use('ggplot')
    plt.rcParams.update({"font.size": 20})
    main()
