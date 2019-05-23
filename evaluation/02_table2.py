"""
Creates a Table 2 using session-level data.
"""
import os
import sys
import nltk
import math
import argparse
import Levenshtein
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
import scipy.spatial.distance
import gensim.downloader as api

import evaluation.util
from evaluation import config
import evaluation.embeddings.util as eeu

METRIC_NAMES = ['WER', 'BLEU', 'COSINE', 'EMD']
SPEAKERS = ['T', 'P']


def main(args):
	"""
	Main function which handles program flow, etc.
	:param args: Argparse namespace.
	"""
	if not os.path.exists(config.META_FQN):
		print(f'File does not exist: {config.META_FQN}')
		sys.exit(1)

	# Load the paired file. We want ALL the data.
	paired = evaluation.util.load_paired_json(skip_empty=True)

	# Compute WER and BLEU here.
	hash2metrics = compute_metrics(paired)

	# For each hash, determine the values for each dimension of interest.
	hash2dim_values, unique_dim_vals = load_dimensions()

	# Create the tree-dictionary of metadata dimensions.
	# Key Structure: speaker, metric_name, dimension, dim_value
	# Values: List of distances (float).
	accumulator: Dict[str, Dict[str, Dict[str, Dict[int, List[float]]]]] = {}
	for speaker in SPEAKERS:
		accumulator[speaker] = {}
		for metric_name in METRIC_NAMES:
			# Create the top level keys for embedding names.
			accumulator[speaker][metric_name]: Dict[str, Dict[int, List[float]]] = {}
			for dim in unique_dim_vals.keys():
				# Create the keys for dimension names.
				accumulator[speaker][metric_name][dim]: Dict[int, List[float]] = {}
				for val in unique_dim_vals[dim]:
					accumulator[speaker][metric_name][dim][val]: List[float] = []

	# Populate the accumulator with our hash2metrics data.
	for hash_ in hash2metrics:
		# For each dimension, find the dimension value for this hash file.
		for dim in config.DIMENSIONS:
			if dim not in hash2dim_values[hash_]:
				continue
			dim_val_for_this_hash = hash2dim_values[hash_][dim]
			for speaker in SPEAKERS:
				for metric in METRIC_NAMES:
					vals = hash2metrics[hash_][speaker][metric]
					accumulator[speaker][metric][dim][dim_val_for_this_hash] += vals

	# For each dimension and unique value, print the mean.
	results_fqn = 'table2_session_metrics.csv'
	with open(results_fqn, 'w') as f:
		header = 'speaker,dim,val'
		for name in METRIC_NAMES:
			header += f',{name}_n,{name}_mean,{name}_std'
		f.write(header + '\n')

		for speaker in SPEAKERS:
			for dim in unique_dim_vals.keys():
				for val in unique_dim_vals[dim]:
					result = f'{speaker},{dim},{val}'
					for name in METRIC_NAMES:
						mean, std, n = get_mean_std(speaker, name, accumulator, dim, val)
						if mean is None:
							result += ',,,'
						else:
							result += f',{n},{mean},{std}'
					f.write(result + '\n')
	print(results_fqn)


def load_dimensions() -> (Dict[str, Dict[str, int]], Dict[str, List[int]]):
	"""
	Loads the metadata file and returns various dimensions for each hash file.
	:return hash2dims: Dictionary with key=hash, value=Dict of dimensions.
	:return unique_dim_vals: For each dimension, contains the unique values.
	"""
	hash2dims = {}
	unique_dim_vals = {x: set() for x in config.DIMENSIONS}
	df = pd.read_csv(config.META_FQN, sep='\t')
	for _, row in df.iterrows():
		hash_ = row['hash']
		# Skip hashes already seen.
		if hash_ in hash2dims:
			print(f'Duplicate: {hash_}')
			continue
		# Populate the dimension's values.
		values = {}
		for key in config.DIMENSIONS:
			v = row[key]
			if str(v) == 'nan':
				continue
			elif isinstance(v, float) or isinstance(v, int):
				v = int(v)
				values[key] = v
				unique_dim_vals[key].add(v)
			elif key == 'gender_imputed':
				values[key] = v
				unique_dim_vals[key].add(v)
		hash2dims[hash_] = values

	return hash2dims, unique_dim_vals


def compute_metrics(paired: Dict) -> Dict:
	"""
	Computes WER and BLEU.
	"""
	# # Load the Word2vec model.
	# print(f'Loading word2vec model...')
	# model = api.load('word2vec-google-news-300')
	# w2v_keys = set(model.vocab)

	# Keys: Hash, speaker, metric_name
	hash2metrics: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
	
	# Concat all sentences for the speaker/patient into a single string.
	# Keys: Hash, speaker. Value: List of phrases.
	gts = {}
	preds = {}
	keys = list(paired.keys())
	for i in tqdm(range(len(keys)), desc='Concatenating'):
		gid = keys[i]
		hash_ = paired[gid]['hash']
		speaker = paired[gid]['speaker']
		if speaker not in SPEAKERS:
			continue

		if hash_ not in gts:
			gts[hash_] = {s: [] for s in SPEAKERS}
			preds[hash_] = {s: [] for s in SPEAKERS}

		gts[hash_][speaker].append(paired[gid]['gt'])
		preds[hash_][speaker].append(paired[gid]['pred'])

	# For each hash, compute the metrics.
	hash_keys = list(gts.keys())
	for i in tqdm(range(len(hash_keys)), desc='Computing Metrics'):
		hash_ = hash_keys[i]
		for speaker in SPEAKERS:
			# Create the single strings.
			gt = ' '.join(gts[hash_][speaker])
			pred = ' '.join(preds[hash_][speaker])

			# Compute WER and BLEU.
			bleu = nltk.translate.bleu_score.sentence_bleu(references=[gt], hypothesis=pred)
			wer = word_error_rate(pred, gt)
			
			# If new hash, populate the dict with accumulator lists.
			if hash_ not in hash2metrics:
				hash2metrics[hash_] = {}
				for s in SPEAKERS:
					hash2metrics[hash_][s] = {name: [] for name in METRIC_NAMES}

			# Compute embedding distances.
			# Error handling to avoid nan, inf, and zeros.
			# emd = model.wmdistance(gt.split(' '), pred.split(' '))  # Requires List[str] of words.
			# if not math.isnan(emd) and emd > 0 and not math.isinf(emd):
			# 	hash2metrics[hash_][speaker]['EMD'].append(emd)

			# gt_embed = eeu.encode_from_dict('word2vec', model, w2v_keys, gt)
			# pred_embed = eeu.encode_from_dict('word2vec', model, w2v_keys, pred)
			# if gt_embed is not None and pred_embed is not None:
			# 	cosine = scipy.spatial.distance.cosine(gt_embed, pred_embed)
			# 	if not math.isnan(cosine) and cosine > 0 and not math.isinf(cosine):
			# 		hash2metrics[hash_][speaker]['COSINE'].append(cosine)

			# Store the result.
			hash2metrics[hash_][speaker]['WER'].append(wer)
			hash2metrics[hash_][speaker]['BLEU'].append(bleu)

	return hash2metrics


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


def get_mean_std(speaker: str, metric_name: str, accumulator: Dict, dim: str, val: int):
	"""
	Checks if the embedding name, dim, val combo is empty or not in `accumulator`.
	:param speaker: P or T
	:param metric_name: WER or BLEU
	:param accumulator: Accumulator dictionary of distance values.
	:param dim: Dimension name.
	:param val: Dimension value.
	:return mean: Mean value of the array.
	:return std: Standard deviation.
	:return n: Number of elements.
	"""
	values = np.asarray(accumulator[speaker][metric_name][dim][val])
	n = len(values)
	if n == 0:
		return None, None, None
	mean = values.mean()
	std = values.std()
	return mean, std, n


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--distance', type=str, choices=['cosine', 'wasserstein'])
	main(parser.parse_args())
