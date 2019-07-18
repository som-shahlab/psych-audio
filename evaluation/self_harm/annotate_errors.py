"""
Loads each of the self-harm examples and asks the user to annotate whether
the ASR transcription is similar to the GT (in terms of phonetics and meaning).
"""
import os
import sys
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from evaluation.self_harm import config


def main():
    """Types of errors

    0: No error. Exact match.s 
    1: Sounds similar, similar meaning
    2: Sounds similar, different meaning
    3: Sounds different, similar meaning
    4: Sounds different, different meaning
    """
    counts = Counter()
    examples = load_examples()
    with open(config.ERROR_RESULT_FQN, "w") as f:
        f.write("eid,error_type\n")
        for i, ex in enumerate(examples):
            error_type = -1
            gt, pred = ex
            # Remove newlines and trailing whitespace.
            gt = gt.strip()
            pred = pred.strip()

            # Check if an exact match.
            if pred == gt:
                error_type = 0
            # One {gt, pred} is blank.
            elif pred == "" or gt == "":
                error_type = 4
            else:
                print(f"--- {i+1} of {len(examples)} ----------------------")
                print(f"GT:\t{gt}")
                print(f"Pred:\t{pred}")
                error_type = ask_user_error_type()

            f.write(f"{i+1},{error_type}\n")
            f.flush()
    print(config.ERROR_RESULT_FQN)


def ask_user_error_type():
    """Asks the user what type of error (phonetic/semantic) from keyboard"""
    sounds_similar = "magic"
    similar_meaning = "magic"
    while sounds_similar not in ["1", "2", ""]:
        sounds_similar = input("Sounds similar? [1=yes] ")
    while similar_meaning not in ["1", "2", ""]:
        similar_meaning = input("Same meaning? [1=yes] ")

    sounds_similar = True if sounds_similar == "1" else False
    similar_meaning = True if similar_meaning == "1" else False

    # Determine error type.
    if sounds_similar and similar_meaning:
        return 1
    elif sounds_similar and not similar_meaning:
        return 2
    elif not sounds_similar and similar_meaning:
        return 3
    elif not sounds_similar and not similar_meaning:
        return 4


def load_examples():
    """Loads the aligned TSV self-harm examples"""
    filenames = os.listdir(config.OUT_DIR)
    examples = []
    for filename in tqdm(filenames):
        fqn = os.path.join(config.OUT_DIR, filename)
        df = pd.read_csv(fqn, sep="\t")
        gt_words = []
        pred_words = []
        for i, row in df.iterrows():
            gt = row["gt"]
            pred = row["pred"]
            if isinstance(gt, float) and math.isnan(gt):
                gt = ""
            if isinstance(pred, float) and math.isnan(pred):
                pred = ""
            gt_words.append(gt)
            pred_words.append(pred)

        gt = " ".join(list(map(str, gt_words)))
        pred = " ".join(list(map(str, pred_words)))
        ex = (gt, pred)
        examples.append(ex)
    return examples


if __name__ == "__main__":
    main()
