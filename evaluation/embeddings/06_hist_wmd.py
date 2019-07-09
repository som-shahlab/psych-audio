"""
Creates the histogram figure which shows the similarity/difference
between randomly generated sentences and sentences from our corpus.
"""
import os
import sys
import math
import argparse
import numpy as np
from typing import *
import scipy.stats
from tqdm import tqdm
import sklearn.preprocessing
import scipy.spatial.distance
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gensim.downloader as api


import evaluation.embeddings.util as eeu
from evaluation import config


def main(args):
    # Generate random sentences.
    random_sentences: List[str] = eeu.random_sentences(args.n, use_corpus=False)

    # Select N sentences from our GT corpus.
    corpus_sentences: List[str] = eeu.random_sentences(args.n, use_corpus=True)

    print(f"Loading the model..")
    model = api.load("word2vec-google-news-300")

    print("Computing random dists...")
    random_dists = eeu.pairwise_wmd(model, random_sentences)
    eeu.print_metrics(random_dists, "WMD-random")

    print(f"Computing corpus dists...")
    corpus_dists = eeu.pairwise_wmd(model, corpus_sentences)
    eeu.print_metrics(corpus_dists, "WMD-corpus")

    print("--------- Unequal Variance --------")
    statistic, pval = scipy.stats.ttest_ind(
        random_dists, corpus_dists, equal_var=False
    )
    print(f"Statistic: {statistic}")
    print(f"Two-Tailed P Value: {pval}")
    print(f"n sentence: {args.n}")
    print(f"n corpus: {len(corpus_dists)}")
    print(f"n random: {len(random_dists)}")

    out_fqn = os.path.join(args.output_dir, "wmd.png")
    eeu.plot_histogram(out_fqn, random_dists, corpus_dists)


if __name__ == "__main__":
    plt.style.use("ggplot")
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Where to save the output figure.")
    parser.add_argument(
        "--n",
        type=int,
        help="Number of sentences to use for pairwise comparison.",
    )
    main(parser.parse_args())
