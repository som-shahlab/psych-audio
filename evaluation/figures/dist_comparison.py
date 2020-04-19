"""This script creates boxplots that compare distance functions (cosine, euclidean, EMD, etc)."""
import os
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

data_dir = "results/"


def main():
    for metric in ["euclidean", "emd", "cosine"]:
        x1 = np.load(os.path.join(data_dir, f"{metric}_random_1000.npy"))
        x2 = np.load(os.path.join(data_dir, f"{metric}_corpus_1000.npy"))
        dual_hist(metric, x1, x2)


def dual_hist(metric: str, arr1: np.ndarray, arr2: np.ndarray, n_bins=50):
    # Implementation quirk: Histograms take only integers. Need to scale our
    # floating point distances up, then re-scale down.
    MULTIPLIER = 10000
    arr1 *= MULTIPLIER
    arr2 *= MULTIPLIER

    # Create the global bins.
    global_min = min(arr1.min(), arr2.min())
    global_max = max(arr1.max(), arr2.max())
    step = (global_max - global_min) / n_bins
    bins = np.arange(global_min, global_max, int(np.ceil(step))).astype(np.int64)

    count1, bin1 = np.histogram(arr1, bins)
    count2, bin2 = np.histogram(arr2, bins)
    count1 = count1 / count1.sum()
    count2 = count2 / count2.sum()
    print(metric, count1.max(), count2.max())

    use_log = False
    plt.figure(figsize=(12, 12))
    # gs1 = gridspec.GridSpec(1, 2)
    # gs1.update(wspace=0, hspace=0)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.grid(b=True, which="major", color="#dddddd", linestyle="-")
    ax2.grid(b=True, which="major", color="#dddddd", linestyle="-")

    # ax1 = plt.subplot(gs1[0])
    y1, x1, _ = ax1.hist(
        arr1,
        bins,
        log=use_log,
        orientation="horizontal",
        density=True,
        facecolor="#E69F00",
    )
    ax1.set_title(f"{metric} random")
    ax1.invert_xaxis()

    # plt.figure(num=None, figsize=(5, 8))
    # plt.title(f"{metric} corpus")
    # ax2 = plt.subplot(gs1[1])
    y2, x2, _ = ax2.hist(
        arr2,
        bins,
        log=use_log,
        orientation="horizontal",
        density=True,
        facecolor="#56B4E9",
    )
    ax2.set_title(f"{metric} corpus")
    hist_max = 1.2 * max(y1.max(), y2.max())
    # ax1.set_xlim(0, hist_max)
    ax2.set_xlim(0, hist_max)
    fqn = f"{metric}.eps"
    # plt.show()
    plt.savefig(fqn, bbox_inches="tight")
    print(fqn)


if __name__ == "__main__":
    main()
