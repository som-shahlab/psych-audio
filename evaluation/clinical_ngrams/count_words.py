"""
This script computes the clinical ngram score between the GT and prediction.
"""
import os
import re
import sys
import argparse
from tqdm import tqdm
from typing import List, Dict
import evaluation.util

# The name of the clinically-relevant words file. Each line should contain a phrase or word that's clinically useful.
RELEVANT_WORDS_FQN = 'evaluation/clinical_ngrams/ngrams.txt'


def main(args):
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
	parser.add_argument('text_file', type=str, help='Location of the combined GT/pred text file.')
	parser.add_argument('speaker', type=str, choices=['T', 'P'], help='Compute for therapist or patient.')
	main(parser.parse_args())
