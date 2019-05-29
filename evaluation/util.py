"""
Utility functions for evaluation.
"""
import json
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
			if len(data[gid]['gt']) == 0 and len(data[gid]['pred']) == 0:
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
	sentence = sentence.lower().strip().replace('\'', '')
	sentence = unidecode.unidecode(sentence)
	return sentence


def word_error_rate(pred: List[str], target: List[str]) -> float:
	"""
	Computes the Word Error Rate, defined as the edit distance between the
	two provided sentences after tokenizing to words.

	:param pred: List of predicted words.
	:param target: List of ground truth words.
	:return:
	"""
	# build mapping of words to integers
	b = set(pred + target)
	word2char = dict(zip(b, range(len(b))))

	# map the words to a char array (Levenshtein packages only accepts strings)
	w1 = [chr(word2char[w]) for w in pred]
	w2 = [chr(word2char[w]) for w in target]

	d = Levenshtein._levenshtein.distance(''.join(w1), ''.join(w2))
	wer = d / max(len(target), 1)
	wer = min(wer, 1.0)
	return wer