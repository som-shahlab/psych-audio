"""
Loads the cropped sentences (tsv files) and computes WER.
"""
import os
import sys
import nltk
import argparse
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

    statistic, pval = scipy.stats.ttest_ind(
        wers["T"], wers["P"], equal_var=False
    )
    print(f"t-Statistic: {statistic}")
    print(f"Two-Tailed P Value: {pval}")


if __name__ == "__main__":
    main()
