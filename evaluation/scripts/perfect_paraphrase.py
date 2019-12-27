"""
Computes the semantic score of a perfect paraphrase pair.
This semantic score can then be used to compute a % paraphrase for ASR sentences.
"""
import os
import sys
from tqdm import tqdm
import gensim.downloader

PPDB_FQN = '/Users/ahaque/Downloads/ppdb-2.0-s-all'

def main():
    #model = gensim.downloader.load("word2vec-google-news-300")
    #w2v_keys = set(model.vocab)
    # Load the dataset.
    examples = load_ppdb()
    print(len(examples))

    # Compute the semantic distance for each example.
    for (p1, p2, score) in examples:
        dist = compute_dist()  # TODO


def load_ppdb():
    examples = []
    pbar = tqdm(total=4551746)
    with open(PPDB_FQN, 'r') as f:
        line = f.readline()
        while line is not None and '|||' in line:
            p1, p2, score = process_line(line.strip())
            examples.append((p1, p2, score))
            line = f.readline()
            pbar.update(1)
    pbar.close()
    return examples

def process_line(line: str):
    """
    Extracts both phrases and the score from a single PPDB line.
    """
    lhs, phrase, paraphrase, features, alignment, entailment = line.split('|||')
    feats = features.split(' ')
    for feat in feats:
        if '=' not in feat:
            continue
        name, val = feat.split('=')
        if name == 'PPDB2.0Score':
            score = val
            break

    return phrase, paraphrase, score

if __name__ == '__main__':
    main()
