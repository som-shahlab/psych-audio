"""
Loads each of the self-harm examples and asks the user to annotate whether
the ASR transcription is similar to the GT (in terms of phonetics and meaning).
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from evaluation.self_harm import config


def main():
    """Types of errors

    0: Sounds similar, similar meaning
    1: Sounds similar, different meaning
    2: Sounds different, similar meaning
    3: Sounds different, different meaning
    """
    counts = Counter()
    examples = load_examples()
    for i, ex in enumerate(examples):
        gt, pred = ex
        print(f"--- {i+1} of {len(examples)} --------------------------")
        print(f"GT:\t{gt}")
        print(f"Pred:\t{pred}")
        error_type = ask_user_error_type()
        counts[error_type] += 1

        print(counts)


def ask_user_error_type():
    """Asks the user what type of error (phonetic/semantic) from keyboard"""
    exact_match = input("Exact match [y/n] ")
    # TODO: Count exact matches.

    # TODO: Better input handling.
    sounds_similar = input("Sounds similar? [y/n] ")
    similar_meaning = input("Same meaning? [y/n] ")
    sounds_similar = True if sounds_similar == "y" else False
    similar_meaning = True if similar_meaning == "y" else False

    # Determine error type.
    if sounds_similar and similar_meaning:
        return 0
    elif sounds_similar and not similar_meaning:
        return 1
    elif not sounds_similar and similar_meaning:
        return 2
    elif not sounds_similar and not similar_meaning:
        return 3


def load_examples():
    """Loads the aligned TSV self-harm examples"""
    filenames = os.listdir(config.OUT_DIR)
    examples = []
    for filename in filenames:
        fqn = os.path.join(config.OUT_DIR, filename)
        df = pd.read_csv(fqn, sep="\t")
        gt = []
        pred = []
        for i, row in df.iterrows():
            if str(row["gt"]) != "nan":
                gt.append(row["gt"])
            if str(df["pred"]) != "nan":
                pred.append(row["pred"])

        gt = " ".join(list(map(str, gt)))
        pred = " ".join(list(map(str, pred)))
        ex = (gt, pred)
        examples.append(ex)
    return examples


if __name__ == "__main__":
    main()
