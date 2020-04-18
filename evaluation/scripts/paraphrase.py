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
import evaluation.embeddings.util as eeu

PPDB_FQN = "/home/ahaque/Git/psych-audio/ppdb/ppdb-2.0-s-all"


def main():
    print(f"Loading the model..")
    model = gensim.downloader.load("word2vec-google-news-300")

    # Load the dataset.
    print(f"Loading PPDB..")
    examples = load_ppdb()

    # Baseline: Non-paraphrase sentences.
    # Create pairs of real sentences from PPDB and compute EMD.
    N = 10000
    idxs = np.random.choice(len(examples), N * 2, replace=False)
    real_dists = []
    real_wers = []
    for i in range(N):
        s1 = examples[idxs[i]][0]
        s2 = examples[idxs[i + N]][0]
        d = model.wmdistance(s1.split(" "), s2.split(" "))
        if math.isnan(d) or d <= 0 or math.isinf(d):
            continue
        real_dists.append(d)
        real_wers.append(evaluation.util.word_error_rate(s1, s2))
    real_dists = np.asarray(real_dists)
    eeu.print_metrics(real_dists, "Real Sentences EMD")
    eeu.print_metrics(real_wers, "Real Sentences WER")

    # Baseline: Paraphrase sentences.
    # Compute EMD between paraphrased sentences.
    paraphrase_dists = []
    paraphrase_wers = []
    i = 0
    for (p1, p2, score) in examples:
        paraphrase_wers.append(evaluation.util.word_error_rate(p1, p2))
        d = model.wmdistance(p1.split(" "), p2.split(" "))
        if math.isnan(d) or d <= 0 or math.isinf(d):
            continue
        paraphrase_dists.append(d)
        i += 1
        if i == N:
            break
    paraphrase_dists = np.asarray(paraphrase_dists)
    eeu.print_metrics(paraphrase_dists, "Paraphrases EMD")
    eeu.print_metrics(paraphrase_wers, "Paraphrases WER")


def load_ppdb():
    examples = []
    pbar = tqdm(total=4551746)
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
