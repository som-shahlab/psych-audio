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
    # Get sentences from our corpus.
    use_corpus = True if args.source == "corpus" else False
    sentences: List[str] = eeu.random_sentences(args.n, use_corpus=use_corpus)

    dists = None
    print(f"Loading word2vec for {args.metric}...")
    if args.metric == "emd":
        model = api.load("word2vec-google-news-300")
        dists = eeu.pairwise_wmd(model, sentences)
    else:
        model, keys = eeu.load_embedding_model("word2vec")
        embeddings = eeu.batch_encode("word2vec", model, keys, sentences)
        dists = eeu.pairwise_metric(embeddings, args.metric)

    fqn = os.path.join(args.output_dir, f"{args.metric}_{args.source}_{args.n}.npy")
    np.save(fqn, dists)
    print(f"Saved: {fqn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric", choices=["euclidean", "cosine", "emd"],
    )
    parser.add_argument("--source", choices=["corpus", "random"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    main(parser.parse_args())
