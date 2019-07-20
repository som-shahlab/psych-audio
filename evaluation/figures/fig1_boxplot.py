"""
Creates the Figure 1 boxplot (WER/EMD by subgroup).
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import evaluation.config

METRICS = ["WER", "EMD"]


def main():
    table2 = pd.read_csv(evaluation.config.TABLE2_FQN, sep="\t")
    # Create the accumulator for all data.
    data = {
        x: {"key": [], "values": [], "group": [], "lr": []} for x in METRICS
    }

    # Load the overall metrics.
    hashes = set(table2["hash"])
    for hash_ in hashes:
        rows = table2[table2["hash"] == hash_]
        for metric in METRICS:
            data[metric]["lr"].append("1")
            data[metric]["group"].append("group0")
            data[metric]["key"].append("Overall")
            data[metric]["values"].append(rows[metric].mean())

    # Load the gender and speaker information.
    for metric in METRICS:
        # Load the gender information.
        male = table2[table2["gender"] == "Male"][metric].values
        female = table2[table2["gender"] == "Female"][metric].values
        data[metric]["key"] += ["male"] * len(male)
        data[metric]["lr"] += ["1"] * len(male)
        data[metric]["group"] += ["group1"] * len(male)
        data[metric]["values"] += list(male)
        data[metric]["lr"] += ["2"] * len(female)
        data[metric]["group"] += ["group1"] * len(female)
        data[metric]["key"] += ["female"] * len(female)
        data[metric]["values"] += list(female)

        # Load the speaker information.
        therapist = table2[table2["speaker"] == "T"][metric].values
        patient = table2[table2["speaker"] == "P"][metric].values
        data[metric]["group"] += ["group2"] * len(patient)
        data[metric]["key"] += ["patient"] * len(patient)
        data[metric]["lr"] += ["1"] * len(patient)
        data[metric]["values"] += list(patient)
        data[metric]["key"] += ["therapist"] * len(therapist)
        data[metric]["group"] += ["group2"] * len(therapist)
        data[metric]["lr"] += ["2"] * len(therapist)
        data[metric]["values"] += list(therapist)

    # Convert WER from decimal to percentage.
    data["WER"]["values"] = list(np.asarray(data["WER"]["values"]) * 100)

    plot("WER", data)
    plot("EMD", data)


def plot(metric: str, data: pd.DataFrame):
    """
    Creates a single plot for a specific metric.
    
    Args:
        metric (str): ''WER' or 'EMD'
    """
    # Create the actual dataframe.
    df = pd.DataFrame(data=data[metric])

    # Plot the histogram.
    sns.set(style="ticks", palette="pastel")
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    sns.catplot(
        data=df,
        x="lr",
        y="values",
        col="group",
        kind="box",
        aspect=0.4,
        palette=sns.color_palette("Blues", n_colors=2),
    )
    # ax.set_xlabel("")
    sns.despine(offset=10, trim=True)
    plt.savefig(f"boxplot_{metric}.png", dpi=300)
    # ax.set(ylim=(0, 0.8))

    # Violin plot.
    # plt.clf()
    # sns.set(style="whitegrid")
    # ax = sns.violinplot(
    #     x="group", y="values", hue="lr", data=df, palette="muted", split=True
    # )
    # ax.set(ylim=(0, 0.8))
    # sns.despine(offset=10, trim=True)
    # plt.savefig("violin.png")


if __name__ == "__main__":
    main()
