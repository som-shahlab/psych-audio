"""
Utility functions for evaluation.
"""
import json
import numpy as np
import unidecode
import Levenshtein
from typing import *
import evaluation.config as config


def load_paired_json(skip_empty=False):
    """
	Loads the paired json file. Optionally, remove sentences that have a blank GT or pred.
	:return paired: Dictionary of gt/pred/gid/etc values.
	"""
    with open(config.PAIRED_FQN) as f:
        data = json.load(f)

    # Remove empty sentences if desired.
    if not skip_empty:
        return data
    else:
        paired = {}
        for gid in data.keys():
            if len(data[gid]["gt"]) == 0 and len(data[gid]["pred"]) == 0:
                continue
            else:
                paired[gid] = data[gid]
        del data
        return paired


def canonicalize(sentence: str) -> str:
    """
	Converts a sentence into standard, canonicalized format by removing
	punctuation, unicode accent characters, etc.
	
	Args:
		sentence (str): Input sentence.
	
	Returns:
		str: Output, cleaned sentence.
	"""
    sentence = sentence.lower().strip().replace("'", "")
    sentence = unidecode.unidecode(sentence)
    sentence = sentence.strip()
    return sentence


def word_error_rate(pred: List[str], target: List[str]) -> float:
    """
	Computes the Word Error Rate, defined as the edit distance between the
	two provided sentences after tokenizing to words.

	:param pred: List of predicted words.
	:param target: List of ground truth words.
	:return:
	"""
    if not isinstance(pred, list) or not isinstance(target, list):
        raise ValueError("word_error_rate: Arguments should be array of words")
    # build mapping of words to integers
    b = set(pred + target)
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts strings)
    w1 = [chr(word2char[w]) for w in pred]
    w2 = [chr(word2char[w]) for w in target]

    d = Levenshtein._levenshtein.distance("".join(w1), "".join(w2))
    wer = d / max(len(target), 1)
    wer = min(wer, 1.0)
    return wer


def print_metrics(label: str, arr: np.ndarray):
    """
    Prints the mean, standard deviation, median and range of an array.
    
    Args:
        label: Label of what the arr contains (e.g., male, therapist)
        arr (np.ndarray): Array of data.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    assert arr.ndim == 1
    mean = arr.mean()
    median = np.median(arr)
    std = arr.std()
    min_ = arr.min()
    max_ = arr.max()
    N = len(arr)
    result1 = f"{label} avg (SD) of {mean:.3f} ({std:.3f}) with"
    result2 = f" (med [rng], {median:.3f} [{min_:.3f}-{max_:.3f}]; n={N})"
    print(result1 + result2)
