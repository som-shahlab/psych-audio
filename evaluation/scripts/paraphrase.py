"""
Computes the semantic score of a perfect paraphrase pair.
This semantic score can then be used to compute a % paraphrase for ASR sentences.
"""
import os
import sys
import math
import numpy as np
from tqdm import tqdm
import gensim.downloader
import evaluation.util
import evaluation.stats
import evaluation.config
import preproc.util
import evaluation.embeddings.util as eeu
from typing import *

# http://paraphrase.org/#/download
# Language: English
# Options: All
# Size: Small
PPDB_FQN = "/home/ahaque/Git/psych-audio/ppdb/ppdb-2.0-s-all"


def main():
    # Takes 3-5 minutes, depending on internet speed.
    print(f"Downloading word2vec model..")
    model = gensim.downloader.load("word2vec-google-news-300")

    # Load the dataset. Takes 30-45 seconds on the small dataset.
    print(f"Loading PPDB..")
    examples = load_ppdb()

    # Baseline: Random words.
    N = 10000
    rand_dists, rand_wers = [], []
    rand_sentences = eeu.random_sentences(N * 5, use_corpus=False)
    idxs = np.random.choice(len(rand_sentences), N * 3, replace=False)
    while len(rand_wers) < N:
        s1 = rand_sentences[idxs[len(rand_wers)]].split(" ")
        s2 = rand_sentences[idxs[len(rand_wers) + N]].split(" ")
        d = model.wmdistance(s1, s2)
        if math.isnan(d) or d <= 0 or math.isinf(d):
            continue
        rand_dists.append(d)
        wer = evaluation.util.word_error_rate(s1, s2)
        rand_wers.append(wer)

    rand_wers = np.asarray(rand_wers) * 100
    eeu.print_metrics(rand_wers, "Random Words WER")
    eeu.print_metrics(rand_dists, "Random Words EMD")

    # Baseline: Random sentences.
    # Create pairs of real sentences from PPDB and compute EMD.
    idxs = np.random.choice(len(examples), N * 5, replace=False)
    real_dists = []
    real_wers = []
    while len(real_wers) < N:
        s1 = examples[idxs[len(real_wers)]][0].split(" ")
        s2 = examples[idxs[len(real_wers) + N]][0].split(" ")
        d = model.wmdistance(s1, s2)
        if math.isnan(d) or d <= 0 or math.isinf(d):
            continue
        real_dists.append(d)
        real_wers.append(evaluation.util.word_error_rate(s1, s2))
    real_dists = np.asarray(real_dists)
    real_wers = np.asarray(real_wers) * 100
    eeu.print_metrics(real_dists, "PPDB Sentences EMD")
    eeu.print_metrics(real_wers, "PPDB Sentences WER")

    # Baseline: Paraphrase sentences.
    # Compute EMD between paraphrased sentences.
    paraphrase_dists = []
    paraphrase_wers = []
    i = 0
    for (p1, p2, _) in examples:
        p1, p2 = p1.split(" "), p2.split(" ")
        paraphrase_wers.append(evaluation.util.word_error_rate(p1, p2))
        d = model.wmdistance(p1, p2)
        if math.isnan(d) or d <= 0 or math.isinf(d):
            continue
        paraphrase_dists.append(d)
        i += 1
        if i == N:
            break
    paraphrase_dists = np.asarray(paraphrase_dists)
    paraphrase_wers = np.asarray(paraphrase_wers) * 100
    eeu.print_metrics(paraphrase_dists, "PPDB Paraphrases EMD")
    eeu.print_metrics(paraphrase_wers, "PPDB Paraphrases WER")


def load_ppdb():
    examples = []
    pbar = tqdm(total=4551746)  # Small dataset: 4551746
    with open(PPDB_FQN, "r") as f:
        line = f.readline()
        while line is not None and "|||" in line:
            pbar.update(1)
            p1, p2, score = process_line(line.strip())
            if p1 is None:
                line = f.readline()
                continue
            examples.append((p1, p2, score))
            line = f.readline()
    pbar.close()
    return examples


def process_line(line: str):
    """
    Extracts both phrases and the score from a single PPDB line.
    """
    lhs, phrase, paraphrase, features, alignment, entailment = line.split("|||")
    phrase = phrase.strip()
    paraphrase = paraphrase.strip()
    if "]" in phrase:
        phrase = remove_brackets(phrase)
    if "]" in paraphrase:
        paraphrase = remove_brackets(paraphrase)
    n_words = len(phrase.split(" "))
    if n_words < 2:
        return None, None, None
    feats = features.split(" ")
    for feat in feats:
        if "=" not in feat:
            continue
        name, val = feat.split("=")
        if name == "PPDB2.0Score":
            score = val
            break

    return phrase, paraphrase, score


def remove_brackets(line: str) -> str:
    """
    Removes brackets from a string.
    Before:
        [SBAR/NP,1] [NP/NP,2] this sort
    After:
        this sort
    """
    start = line.rfind("]")
    return line[start:]


if __name__ == "__main__":
    main()
