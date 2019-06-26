"""
This file computes the BERT embeddings for GT and predictions.

Prerequisites:
- Transcriptions text file (text.txt) resulting from `evaluation/01_extract_phrases.py`.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import evaluation.util
from evaluation import config
from bert_serving.client import BertClient


def main(args):
    if not os.path.exists(config.NPZ_DIR):
        os.makedirs(config.NPZ_DIR)

    print("Creating BERT client...")
    bc = BertClient(check_length=False)

    # Load the paired json file.
    paired = evaluation.util.load_paired_json(skip_empty=True)

    # Optionally, use a subset.
    if args.n_lines > 0:
        subset = {}
        for i, gid in enumerate(paired.keys()):
            if i >= args.n_lines:
                break
            subset[gid] = paired[gid]
        del paired
        paired = subset

    # The BERT client uses lists for everything.
    # Convert `paired` into list form.
    pred_sentences, gt_sentences, gids = [], [], []
    for gid in paired.keys():
        gids.append(gid)
        gt_sentences.append(paired[gid]["gt"])
        pred_sentences.append(paired[gid]["pred"])

    # Get embeddings.
    print(f"Encoding {len(gids)} sentences...")
    # Size: (N, SEQ_LEN, F)
    # N: Number of sentences
    # SEQ_LEN: Max sentence length.
    # F: BERT feature size.
    pred_embeddings = bc.encode(pred_sentences)
    gt_embeddings = bc.encode(gt_sentences)

    print("Converting text to numpy...")
    pred_sentences = np.asarray(pred_sentences)
    gt_sentences = np.asarray(gt_sentences)

    # Write embeddings to file.
    print("Saving embeddings...")
    pred_fqn = os.path.join(config.NPZ_DIR, "bert_pred.npz")
    gt_fqn = os.path.join(config.NPZ_DIR, "bert_gt.npz")
    np.savez_compressed(
        pred_fqn, embeddings=pred_embeddings, gids=gids, text=pred_sentences
    )
    np.savez_compressed(
        gt_fqn, embeddings=gt_embeddings, gids=gids, text=gt_sentences
    )
    print(pred_fqn)
    print(gt_fqn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_lines",
        type=int,
        default=-1,
        help="Number of lines to process. Use -1 to process all.",
    )
    main(parser.parse_args())
