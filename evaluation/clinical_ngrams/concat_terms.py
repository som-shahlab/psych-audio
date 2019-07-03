"""
Concats all words for a PHQ bucket. Used for generating Table 3.
"""
import os
import sys
import pandas as pd
import evaluation.config as config


def main():
    df = pd.read_csv(config.PHQ_TERM_FQN, sep="\t")
    terms = {x: [] for x in range(1, 10)}
    for i, row in df.iterrows():
        phq = int(row["PHQ"])
        ngram = row["TERM"]
        terms[phq].append(ngram)

    for i in range(1, 10):
        print(i, ", ".join(terms[i]))


if __name__ == "__main__":
    main()
