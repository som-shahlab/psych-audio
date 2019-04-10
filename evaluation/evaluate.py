"""
This file contains functions for evaluating the Google Speech API transcription performance.
"""
import os
import sys
import json
import nltk
import argparse
import Levenshtein
import numpy as np
from typing import List, Dict, Set


def main(args):
	if not os.path.exists(args.machine_dir):
		print(f'Path does not exist: {args.machine_dir}')
		sys.exit(0)
	if not os.path.exists(args.gt_dir):
		print(f'Path does not exist: {args.gt_dir}')
		sys.exit(0)

	with open('results.csv', 'w') as f:
		for filename in os.listdir(args.machine_dir):
			hash = filename.replace('.json', '')
			# Load and standardize the ground truth.
			machine = load_json(os.path.join(args.machine_dir, filename))
			gt = load_json(os.path.join(args.gt_dir, filename))

			# Split the machine and GT into phrases, according to the GT timestamps.
			buckets = sorted(np.unique(gt['timestamps']))
			gt_phrases = get_phrases(buckets, gt)
			machine_phrases = get_phrases(buckets, machine)

			# Determine the speaker of each phrase.
			phrase_speakers = {}
			for ts in buckets:
				idx = gt['timestamps'].index(ts)
				speaker = gt['speakerTags'][idx]
				phrase_speakers[ts] = speaker

			# See: https://stackoverflow.com/questions/40542523/nltk-corpus-level-bleu-vs-sentence-level-bleu-score
			list_of_hypotheses = []
			list_of_references = []

			for ts in buckets:
				# Compose the reference and hypothesis.
				r = gt_phrases[ts]
				h = machine_phrases[ts]
				references = [r]
				list_of_hypotheses.append(h)
				list_of_references.append(references)

				# Compute metrics.
				bleu1 = nltk.translate.bleu_score.sentence_bleu(references=references, hypothesis=h)
				speakerTag = phrase_speakers[ts]

				# Write to file.
				word_error_rate = wer(h, r)
				result = f'{hash},{ts},{speakerTag},{bleu1},{word_error_rate}\n'
				f.write(result)


def get_phrases(buckets: List, data: Dict) -> Dict[float, List[str]]:
	"""
	Returns a dictionary of (timestamp, words) where words contains the phrases spoken at `timestamp`.
	:param buckets: Python list of bucket timestamps (from GT).
	:param data: Dictionary of timestamps and words.
	:return: Dictionary of phrases.
	"""
	idx = 0  # Bucket index.
	phrases = {x: [] for x in buckets}
	for i in range(len(data['timestamps'])):
		ts, word = data['timestamps'][i], data['words'][i]
		if idx + 1 < len(buckets) and ts >= buckets[idx + 1]:
			idx += 1
		phrases[buckets[idx]].append(word)
	return phrases


def load_json(fqn: str):
	"""
	Loads a json transcription file.

	A = json.load(fqn)
	A['results'] is a Python list of dictionaries.
	B = A['results'][0]
	B['alternatives'] is a Python list of dictionaries.
	C = B['alternatives'][0] is a dictionary of transcription results.
		C Keys: transcript, confidence, words
	C['words'] is a Python list of dictionaries.
	D = C['words'][0] contains the transcription for a single word.
		D Keys: startTime, endTime, word, confidence, speakerTag

	:param fqn: Path to the machine-generated transcription.
	:return:
	"""
	with open(fqn, 'r') as f:
		A = json.load(f)

	timestamps = []
	words = []
	speakerTags = []
	# For each word, add it to our list.
	for B in A['results']:
		for C in B['alternatives']:
			for D in C['words']:
				# Get the core content.
				startTime = float(D['startTime'].replace('s', ''))
				timestamps.append(startTime)
				words.append(D['word'])

				# Add the speaker information.
				if 'speakerTag' in D:
					speakerTags.append(D['speakerTag'])
				else:
					speakerTags.append('')

	result = {'timestamps': timestamps, 'words': words, 'speakerTags': speakerTags}
	return result


def wer(pred: List[str], target: List[str]) -> float:
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

	d = Levenshtein.distance(''.join(w1), ''.join(w2))
	wer = d / len(target)
	wer = min(wer, 1.0)
	return wer


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('machine_dir', type=str, help='Location of the predicted json transcriptions.')
	parser.add_argument('gt_dir', type=str, help='Location of the ground truth json transcription.')
	main(parser.parse_args())
