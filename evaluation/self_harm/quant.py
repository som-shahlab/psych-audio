"""
Loads the cropped sentences (tsv files) and computes WER.
"""
import os
import sys
import nltk
import argparse
from typing import *
import scipy.stats
import numpy as np
import pandas as pd
import preproc.util
import evaluation.stats
import evaluation.util
import gensim.downloader as api
from evaluation.self_harm import config
import evaluation.embeddings.util as eeu


def main():
    # Load the W2V model.
    print(f"Loading word2vec (takes 2-3 minutes)..")
    model = api.load("word2vec-google-news-300")

    # Load the self-harm data.
    print("Computing corpus-level results... (1-2 minutes)")
    harm_wers, harm_emds = compute_self_harm_metrics(model)

    # Load the corpus-level data.
    df = pd.read_csv(evaluation.config.TABLE2_FQN, sep="\t")
    corpus_wers = df["WER"].values
    corpus_emds = df["EMD"].values

    print("------ Self-Harm WER ------")
    evaluation.util.print_metrics("Therapist", harm_wers["T"])
    evaluation.util.print_metrics("Patient", harm_wers["P"])
    evaluation.util.print_metrics("Aggregate", harm_wers["All"])
    evaluation.util.print_metrics("Corpus", corpus_wers)

    print("------ Self-Harm EMD ------")
    evaluation.util.print_metrics("Therapist", harm_emds["T"])
    evaluation.util.print_metrics("Patient", harm_emds["P"])
    evaluation.util.print_metrics("Aggregate", harm_emds["All"])
    evaluation.util.print_metrics("Corpus", corpus_emds)

    # Compute self-harm sentences vs corpus.
    print("------ Self-Harm vs Corpus ------")
    evaluation.stats.difference_test(
        ["WER Self-Harm", "WER Corpus"], harm_wers["All"], corpus_wers
    )
    evaluation.stats.difference_test(
        ["EMD Self-Harm", "EMD Corpus"], harm_emds["All"], corpus_emds
    )

    print("------ Therapist Self-Harm vs Patient Self-Harm ------")
    # Compute self-harm patient vs therapist.
    evaluation.stats.difference_test(
        ["WER Therapist", "WER Patient"], harm_wers["T"], harm_wers["P"]
    )
    evaluation.stats.difference_test(
        ["EMD Therapist", "EMD Patient"], harm_emds["T"], harm_emds["P"]
    )


def compute_self_harm_metrics(model):
    """
    Computes self-harm WER and EMD.

    Args:
        model: Gensim model.

    Returns:
        wers: Dictionary containing keys 'T', 'P', 'All'. Each dictionary
            value is a python list of numbers, which correspond to a data
            point for WER (e.g., one sentence).
        emds: Same as `wers` but for earth mover distance (EMD).
    """
    # Load the EID -> speaker file.
    eid2speaker = {}
    with open(config.SPEAKER_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            eid, speaker = line.strip().split(" ")
            eid2speaker[int(eid)] = speaker

    # Load the cropped files.
    filenames = os.listdir(config.OUT_DIR)

    # Compute metrics for patient vs therapist.
    wers = {"T": [], "P": [], "All": []}
    emds = {"T": [], "P": [], "All": []}
    for filename in filenames:
        fqn = os.path.join(config.OUT_DIR, filename)
        df = pd.read_csv(fqn, sep="\t")
        eid = int(filename[:3])
        speaker = eid2speaker[eid]

        # Compose the GT and pred sentences.
        gt = []
        pred = []
        conf = []
        for _, row in df.iterrows():
            gt_word = row["gt"]
            pred_word = row["pred"]

            # Check if nan.
            if isinstance(gt_word, float):
                gt_word = ""
            if isinstance(pred_word, float):
                pred_word = ""

            gt.append(preproc.util.canonicalize_word(gt_word))
            pred.append(pred_word)
            conf.append(row["conf"])

        gt = " ".join(gt)
        pred = " ".join(pred)

        # Compute WER.
        wer = evaluation.util.word_error_rate(pred, gt)
        wers[speaker].append(wer)
        wers["All"].append(wer)

        # Compute EMD.
        _, emd = eeu.compute_distances(model, set(model.vocab), gt, pred)
        if eeu.is_valid_distance(emd):
            emds[speaker].append(emd)
            emds["All"].append(emd)

    for k, v in wers.items():
        wers[k] = np.asarray(v)

    return wers, emds


if __name__ == "__main__":
    main()
