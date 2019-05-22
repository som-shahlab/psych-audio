"""
This script computes the clinical ngram score between the GT and prediction.
"""
import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from collections import Counter

import evaluation.util
from evaluation import config

SPEAKERS = ['T', 'P']
OUTCOMES = ['TP', 'FP', 'FN', 'TN']


def main(args):
	if not os.path.exists(config.META_FQN):
		print(f'File does not exist: {config.META_FQN}')
		sys.exit(1)

	# Load the paired file. We want ALL the data.
	paired = evaluation.util.load_paired_json(skip_empty=False)

	# Load the term file.
	term2phq = load_terms()

	# Results counter.
	counts = {term: {key: 0 for key in OUTCOMES} for term in term2phq}

	# For each segment, compute TP, TN, etc. for each term.
	i = 0
	gid_keys = list(paired.keys())
	for gid in tqdm(gid_keys, desc='Computing Metrics'):
		# Get the data.
		i+=1
		hash_ = paired[gid]['hash']
		gt = paired[gid]['gt']
		pred = paired[gid]['pred']
		speaker = paired[gid]['speaker']

		# Only comptue results for the patient.
		if speaker != 'P':
			continue

		# Skip blank sentences.
		if len(gt) == 0 and len(pred) == 0:
			continue

		# For each term, check if this sentence contains it.
		for term in term2phq:
			in_gt = True if term in gt else False
			in_pred = True if term in pred else False

			if in_gt and in_pred:
				counts[term]['TP'] += 1
			elif in_gt and not in_pred:
				counts[term]['FN'] += 1
			elif not in_gt and in_pred:
				counts[term]['FP'] += 1
			elif not in_gt and not in_pred:
				counts[term]['TN'] += 1

	# Write to file.
	out_fqn = 'table3.tsv'
	with open(out_fqn, 'w') as f:
		f.write(f'phq\tterm\ttp\tfn\tfp\ttn\n')
		for term in counts:
			phq = term2phq[term]
			tp = counts[term]['TP']
			fn = counts[term]['FN']
			fp = counts[term]['FP']
			tn = counts[term]['TN']
			f.write(f'{phq}\t{term}\t{tp}\t{fn}\t{fp}\t{tn}\n')
	print(out_fqn)

def load_terms() -> Dict[str, int]:
	"""
	Loads the PHQ term file.

	Returns:
		term2phq: Dictionary with key=term and value=PHQ number.
	"""
	df = pd.read_csv(config.PHQ_TERM_FQN, sep='\t')
	term2phq: Dict[str, int] = {}
	for _, row in df.iterrows():
		term = row['TERM']
		phq = row['PHQ']
		term2phq[term] = phq
	return term2phq
	

def main2(args):
	# Load the GID->speaker map.
	gid2speaker = evaluation.util.load_gid2speaker()

	# Load the n-grams.
	ngrams = load_ngrams()
	all_grams = sum(ngrams.values(), [])
	Ns: List[int] = sorted(list(ngrams.keys()))

	# Load the GT and prediction file.
	if not os.path.exists(args.text_file):
		print(f'File does not exist: {args.text_file}')
		sys.exit(1)

	sentences: Dict[str, List[str]] = {'gt': [], 'pred': []}
	gids: Dict[str, List[str]] = {'gt': [], 'pred': []}
	with open(args.text_file, 'r') as f:
		lines = f.readlines()

	# Store the sentences and the global ID.
	for i in range(len(lines)):
		gid, sentence = tuple(lines[i].strip().split(','))
		key = 'pred' if i % 2 == 0 else 'gt'
		sentences[key].append(sentence)
		gids[key].append(gid)

	# Compute frequencies per gram.
	metric_names = ['TP', 'FP', 'TN', 'FN']
	# For each gram, create a running total of metrics.
	metrics = {y: {x: 0 for x in metric_names} for y in all_grams}

	# For each sentence, compute the TP, TN, etc. rates.
	for gid in tqdm(range(len(sentences['gt'])), desc='Processing'):
		speaker = gid2speaker[gid]
		if speaker != args.speaker:
			continue
		gt = sentences['gt'][gid]
		pred = sentences['pred'][gid]

		# Compute metrics for each n-gram. N refers to the `n` in ngram.
		for n in ngrams.keys():
			# Compute total number of ngrams in this sentence.
			n_ngrams = max(len(gt.split(' ')) + 1 - n, 0)
			# Get all ngrams for this N.
			grams = ngrams[n]
			for gram in grams:
				# Find all occurrences of this ngram in the GT and pred.
				n_gt = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(gram), gt))
				n_pred = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(gram), pred))

				# Count TP/TN/FP/FN.
				tp = min(n_gt, n_pred)
				fp, fn = 0, 0
				if n_gt > n_pred:
					fn = n_gt - tp
				elif n_gt < n_pred:
					fp = n_pred - tp
				tn = n_ngrams - tp - fn - fp

				# Increment the TP/TN/FN/FP counters.
				metrics[gram]['TP'] += tp
				metrics[gram]['TN'] += tn
				metrics[gram]['FN'] += fn
				metrics[gram]['FP'] += fp

	# Write metrics to file.
	filename = 'therapist.csv' if args.speaker == 'T' else 'patient.csv'
	with open(filename, 'w') as f:
		f.write(f'n,gram,tp,tn,fp,fn' + '\n')
		for gram in metrics:
			n = len(gram.split(' '))
			tp = metrics[gram]['TP']
			tn = metrics[gram]['TN']
			fn = metrics[gram]['FN']
			fp = metrics[gram]['FP']
			line = f'{n},{gram},{tp},{tn},{fn},{fp}'
			f.write(line + '\n')


def load_ngrams() -> Dict[int, List[str]]:
	"""
	Loads the clinically-relevant words file and returns the uni, bi, tri, etc. grams.
	:return ngrams: Dictionary of uni, bi, trigrams, etc.
	"""
	ngrams: Dict[int, List[str]] = {}
	with open(RELEVANT_WORDS_FQN, 'r') as f:
		lines = f.readlines()

	for line in lines:
		gram = line.strip().lower()
		n_tokens = len(gram.split(' '))
		if n_tokens not in ngrams:
			ngrams[n_tokens]: List[str] = []
		# Make sure the gram is not already in the dictionary.
		if gram not in ngrams[n_tokens]:
			ngrams[n_tokens].append(gram)

	return ngrams


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	main(parser.parse_args())
