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
import evaluation.util
from evaluation.self_harm import config
import evaluation.embeddings.util as eeu


def main():
    # Load the W2V model.
    # print(f"Loading the model..")
    # model, keys = eeu.load_embedding_model("word2vec")

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
        # embeddings = eeu.batch_encode("word2vec", model, keys, [gt, pred])
        # print(embeddings)

    for k, v in wers.items():
        wers[k] = np.asarray(v)

    print("Therapist")
    evaluation.util.print_metrics(wers["T"])

    print("Patient")
    evaluation.util.print_metrics(wers["P"])

    print("All")
    evaluation.util.print_metrics(wers["All"])
    N = len(wers["All"])

    # Compare to randomly selected sentences from the corpus.
    # Select random N paired examples from the corpus.
    corpus_wers = get_random_corpus_wers(N)
    print("Corpus WERs")
    evaluation.util.print_metrics(corpus_wers)

    print("Corpus vs Self-Harm")
    statistic, pval = scipy.stats.ttest_ind(
        wers["All"], corpus_wers, equal_var=False
    )
    print(f"t-Statistic: {statistic}")
    print(f"Two-Tailed P Value: {pval}")


def get_random_corpus_wers(N: int):
    """
    Loads the paired.json file, selects N examples randomly, then computes WER.
    
    Args:
        N (int): Number of sentences to select.
    """
    # Load the paired file.
    paired = evaluation.util.load_paired_json(skip_empty=True)

    # Select random examples.
    keys = list(paired.keys())
    ridxs = np.random.choice(len(keys), N, replace=False)

    # Compute WERs.
    wers = np.zeros((N,))
    for i, ridx in enumerate(ridxs):
        gid = keys[ridx]
        pred = paired[gid]["pred"]
        gt = paired[gid]["gt"]
        wers[i] = evaluation.util.word_error_rate(pred, gt)

    return wers


if __name__ == "__main__":
    main()
