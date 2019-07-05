"""
This script computes the clinical ngram score between the GT and prediction.
"""
import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from collections import Counter

import evaluation.util
from evaluation import config

SPEAKERS = ["T", "P"]
OUTCOMES = ["TP", "FP", "FN", "TN", "P"]


def main(args):
    if not os.path.exists(config.META_FQN):
        print(f"File does not exist: {config.META_FQN}")
        sys.exit(1)

    # Load the paired file. We want ALL the data.
    paired = evaluation.util.load_paired_json(skip_empty=False)

    # Load the term file.
    term2phq = load_terms()

    # Results counter.
    counts = {term: {key: 0 for key in OUTCOMES} for term in term2phq}

    # For each segment, compute TP, TN, etc. for each term.
    i = 0
    gid_keys = list(paired.keys())
    for gid in tqdm(gid_keys, desc="Computing Metrics"):
        # Get the data.
        i += 1
        hash_ = paired[gid]["hash"]
        gt = paired[gid]["gt"]
        pred = paired[gid]["pred"]
        speaker = paired[gid]["speaker"]

        # Only comptue results for the patient.
        if speaker != "P":
            continue

        # Skip blank sentences.
        if len(gt) == 0 and len(pred) == 0:
            continue

        # For each term, count how many times it appears in this sentence.
        for term in term2phq:
            n_words = len(term.replace("-", "").split(" "))
            n_ngrams = max(len(gt.split(" ")) + 1 - n_words, 0)

            n_gt = sum(1 for _ in re.finditer(r"\b%s\b" % re.escape(term), gt))
            n_pred = sum(
                1 for _ in re.finditer(r"\b%s\b" % re.escape(term), pred)
            )

            # Count TP/TN/FP/FN.
            tp = min(n_gt, n_pred)
            fp, fn = 0, 0
            if n_gt > n_pred:
                fn = n_gt - tp
            elif n_gt < n_pred:
                fp = n_pred - tp
            tn = n_ngrams - tp - fn - fp
            counts[term]["P"] += tp + fn
            counts[term]["TP"] += tp
            counts[term]["FN"] += fn
            counts[term]["FP"] += fp
            counts[term]["TN"] += tn

    # Write to file.
    with open(evaluation.config.TABLE3_FQN, "w") as f:
        f.write(f"phq\tterm\tpos\ttp\tfn\tfp\ttn\n")
        for term in counts:
            phq = term2phq[term]
            pos = counts[term]["P"]
            tp = counts[term]["TP"]
            fn = counts[term]["FN"]
            fp = counts[term]["FP"]
            tn = counts[term]["TN"]
            f.write(f"{phq}\t{term}\t{pos}\t{tp}\t{fn}\t{fp}\t{tn}\n")
    print(evaluation.config.TABLE3_FQN)

    # Print terms with # pos > 0.
    phq2term = {x: [] for x in range(1, 10)}
    for term in term2phq:
        if counts[term]["P"] > 0:
            phq = term2phq[term]
            phq2term[phq].append(term)
    for phq in phq2term:
        keywords = sorted(phq2term[phq])
        print(phq, ", ".join(keywords))


def load_terms() -> Dict[str, int]:
    """
	Loads the PHQ term file.

	Returns:
		term2phq: Dictionary with key=term and value=PHQ number.
	"""
    df = pd.read_csv(config.PHQ_TERM_FQN, sep="\t")
    term2phq: Dict[str, int] = {}
    for _, row in df.iterrows():
        term = row["TERM"]
        phq = row["PHQ"]
        term2phq[term] = phq
    return term2phq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
