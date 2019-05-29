"""
Creates a Table 2 using session-level data.
"""
import os
import sys
import nltk
import math
import time
import argparse
import Levenshtein
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from queue import Queue
from threading import Thread
import scipy.spatial.distance
import gensim.downloader as api

import evaluation.util
from evaluation import config
import evaluation.embeddings.util as eeu

METRIC_NAMES = ['WER', 'BLEU', 'COSINE', 'EMD', 'RAND_WER', 'RAND_BLEU', 'RAND_COSINE', 'RAND_EMD']
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
	hash2metrics = compute_metrics(args, paired)

	# For each hash, determine the values for each dimension of interest.
	hash2dim_values, _ = load_dimensions()

	# For each hash, output its metrics along with its relevant metadata.
	out_fqn = 'table2.tsv'
	with open(out_fqn, 'w') as f:
		# Write the header.
		f.write('hash\tspeaker\tgender\tsess_num\tage\tphq')
		for metric in METRIC_NAMES:
			f.write(f'\t{metric}')
		f.write('\n')

		# for each hash, write the metrics to file.
		for hash_ in hash2metrics:
			for speaker in hash2metrics[hash_]:
				f.write(f'{hash_}\t{speaker}')
				# Some hashes don't have all metadata.
				for dim in config.DIMENSIONS:
					meta_for_this_hash = hash2dim_values[hash_]
					if dim in meta_for_this_hash:
						val = meta_for_this_hash[dim]
					else:
						val = ''
					f.write(f'\t{val}')
						
				for metric in METRIC_NAMES:
					value = hash2metrics[hash_][speaker][metric]
					f.write(f'\t{value}')
				f.write('\n')
	print(out_fqn)
	sys.exit(0)


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


def compute_metrics(args, paired: Dict) -> Dict:
	"""
	Computes WER and BLEU.

	Args:
		paired: Paired dictionary.
		no_embedding: If True, does not compute embedding metrics.
	"""
	# Load the Word2vec model.
	model, w2v_keys = None, None
	if not args.no_embedding:
		print(f'Loading word2vec model...')
		model = api.load('word2vec-google-news-300')
		w2v_keys = set(model.vocab)

	# Keys: Hash, speaker, metric_name
	hash2metrics: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
	
	# Create the workload queue.
	q = Queue()
	r = Queue()

	# Concat all sentences for the speaker/patient into a single string.
	# Keys: Hash, speaker. Value: List of phrases.
	gts = {}
	preds = {}
	added_to_queue = set()
	for gid in paired.keys():
		hash_ = paired[gid]['hash']
		speaker = paired[gid]['speaker']
		if speaker not in SPEAKERS:
			continue

		if hash_ not in gts:
			gts[hash_] = {s: [] for s in SPEAKERS}
			preds[hash_] = {s: [] for s in SPEAKERS}

		gts[hash_][speaker].append(paired[gid]['gt'])
		preds[hash_][speaker].append(paired[gid]['pred'])

		# Add this (hash, speaker) to the workload queue.
		item = (hash_, speaker)
		if item not in added_to_queue:
			q.put(item)
			added_to_queue.add(item)

	n_elements = q.qsize()
	threads = []
	# Create the threads.
	for _ in range(args.n_threads):
		t = Thread(target=worker, args=(args, q, r, model, w2v_keys, gts, preds))
		threads.append(t)

	# Start the threads.
	for t in threads:
		t.start()

	print(f'Waiting for {args.n_threads} threads to finish...')
	pbar_proc = Thread(target=progress_worker, args=(q, n_elements))
	pbar_proc.start()

	# Wait for them to finish.
	for t in threads:
		t.join()

	# Read the result queue and populate our hash2metrics data structure.
	while not r.empty():
		hash_, speaker, metric, value = r.get()
		
		# If new hash, populate the dict with accumulator lists.
		if hash_ not in hash2metrics:
			hash2metrics[hash_] = {}
			for s in SPEAKERS:
				hash2metrics[hash_][s] = {name: 0 for name in METRIC_NAMES}
	
		hash2metrics[hash_][speaker][metric] = value

	return hash2metrics


def progress_worker(q: Queue, max_size: int):
	"""Updates the progress bar."""
	pbar = tqdm(total=max_size)
	prev_count = 0
	while q.qsize() > 0:
		remaining = q.qsize()
		completed = max_size - remaining
		if completed > prev_count:
			delta = completed - prev_count
			prev_count = completed
			pbar.update(delta)
		time.sleep(0.1)


def worker(args, q: Queue, r: Queue, model: Dict, w2v_keys: Set, gts: Dict, preds: Dict):
	"""Multi-threaded implementation of the metrics computation."""
	while not q.empty():
		hash_, speaker = q.get()
		# Create the single strings.
		gt = ' '.join(gts[hash_][speaker])
		pred = ' '.join(preds[hash_][speaker])
		random_sentence = eeu.random_sentences(1, use_corpus=False)[0]

		# Compute WER and BLEU.
		bleu = nltk.translate.bleu_score.sentence_bleu(references=[gt], hypothesis=pred)
		wer = evaluation.util.word_error_rate(pred, gt)
		rand_bleu = nltk.translate.bleu_score.sentence_bleu(references=[gt], hypothesis=random_sentence)
		rand_wer = evaluation.util.word_error_rate(random_sentence, gt)

		# Compute embedding distances.
		n_segments = len(gts[hash_][speaker])
		accumulator = {
			'EMD': [],
			'COSINE': [],
			'RAND_EMD': [],
			'RAND_COSINE': [],
		}

		if not args.no_embedding:
			for j in range(n_segments):
				# Compare the GT vs pred.
				segment_gt = gts[hash_][speaker][j]
				segment_pred = preds[hash_][speaker][j]
				cosine, emd = compute_distances(model, w2v_keys, segment_gt, segment_pred)
				if is_valid_distance(emd):
					accumulator['EMD'].append(emd)
				if is_valid_distance(cosine):
					accumulator['COSINE'].append(cosine)

				# Generate a random sentence and measure the performance of a random sentence.
				random_sentence = eeu.random_sentences(1, use_corpus=False)[0]
				rand_cosine, rand_emd = compute_distances(model, w2v_keys, segment_gt, random_sentence)
				if is_valid_distance(rand_emd):
					accumulator['RAND_EMD'].append(rand_emd)
				if is_valid_distance(rand_cosine):
					accumulator['RAND_COSINE'].append(rand_cosine)

			# Compute the session-level mean cosine and EMD.
			for metric in ['EMD', 'COSINE', 'RAND_EMD', 'RAND_COSINE']:
				avg = np.asarray(accumulator[metric]).mean()
				r.put((hash_, speaker, metric, avg))

		# Store the result.
		r.put((hash_, speaker, 'WER', wer))
		r.put((hash_, speaker, 'BLEU', bleu))
		r.put((hash_, speaker, 'RAND_WER', rand_wer))
		r.put((hash_, speaker, 'RAND_BLEU', rand_bleu))


def compute_distances(model: Dict, keys: Set, sentence1: str, sentence2: str) -> (float, float):
	"""Computes EMD and cosine distance."""
	# EMD requires List[str] of words.
	emd = model.wmdistance(sentence1.split(' '), sentence2.split(' '))

	# Exctract embeddings.
	embed1 = eeu.encode_from_dict('word2vec', model, keys, sentence1)
	embed2 = eeu.encode_from_dict('word2vec', model, keys, sentence2)

	# Cosine distance.
	cosine = None
	if embed1 is not None and embed2 is not None:
		cosine = scipy.spatial.distance.cosine(embed1, embed2)

	return cosine, emd


def is_valid_distance(number: float):
	"""Checks if the number is a valid (non inf, nan, etc.) distance."""
	if number is None:
		return False
	if math.isnan(number):
		return False
	if math.isinf(number):
		return False
	if number < 0:
		return False
	return True


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
	parser.add_argument('--n_threads', default=4, type=int, help='Number of threads to use.')
	parser.add_argument('--no_embedding', action='store_true', help='If True, does not compute embeddings.')
	main(parser.parse_args())
	