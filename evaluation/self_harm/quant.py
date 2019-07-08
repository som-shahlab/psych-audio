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
    w2v_keys = set(model.vocab)

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
        _, emd = eeu.compute_distances(model, w2v_keys, gt, pred)
        if eeu.is_valid_distance(emd):
            emds[speaker].append(emd)
            emds["All"].append(emd)

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
    corpus_wers, corpus_emds = get_random_corpus_wers_emds(model, N)
    print("Corpus WERs")
    evaluation.util.print_metrics(corpus_wers)
    print("Corpus EMDs")
    evaluation.util.print_metrics(corpus_emds)

    evaluation.stats.difference_test(
        ["WER Self-Harm", "WER Corpus"], wers["All"], corpus_wers
    )
    evaluation.stats.difference_test(
        ["EMD Self-Harm", "EMD Corpus"], emds["All"], corpus_emds
    )


def get_random_corpus_wers_emds(model, N: int):
    """
    Loads the paired.json file, selects N examples randomly,
    then computes WER and EMD.
    
    Args:
        model: Gensim model for computing EMD.
        N (int): Number of sentences to select.
    """
    w2v_keys = set(model.vocab)

    # Load the paired file.
    paired = evaluation.util.load_paired_json(skip_empty=True)

    # Select random examples.
    keys = list(paired.keys())

    # Compute WERs.
    wers = np.zeros((N,))
    emds = np.zeros((N,))
    idx = 0
    # Sometimes the predicted sentence is a blank sentence and will cause
    # EMD to be nan. Therefore, keep selecting a differnt one.
    while idx < N:
        # Select a random gid.
        ridx = np.random.choice(len(keys), 1)[0]
        gid = keys[ridx]

        # Get GT and pred.
        pred = paired[gid]["pred"]
        gt = paired[gid]["gt"]

        # Compute WER and EMD.
        wer = evaluation.util.word_error_rate(pred, gt)
        _, emd = eeu.compute_distances(model, w2v_keys, gt, pred)

        # If EMD is valid, store results and continue.
        if eeu.is_valid_distance(emd):
            wers[idx] = wer
            emds[idx] = emd
            idx += 1

    return wers, emds


if __name__ == "__main__":
    main()
