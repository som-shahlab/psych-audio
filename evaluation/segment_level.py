"""
Computes segment-level metrics.
A segment is defined as a single speaker, across multiple timestamps.
"""
import os
import sys
import json
import nltk
import argparse
import Levenshtein
import numpy as np
from tqdm import tqdm
from typing import List, Dict

import preproc.util
import evaluation.gleu


def main(args):
	if not os.path.exists(args.machine_dir):
		print(f'Path does not exist: {args.machine_dir}')
		sys.exit(0)
	if not os.path.exists(args.gt_dir):
		print(f'Path does not exist: {args.gt_dir}')
		sys.exit(0)

	results_file = open('metrics.csv', 'w')
	gt_out_file = open('gt_out.csv', 'w')
	machine_out_file = open('machine_out.csv', 'w')

	results_file.write('hash,seg_ts,seg_tag,bleu,gleu,wer\n')
	ls = sorted(os.listdir(args.machine_dir))
	for i in tqdm(range(len(ls))):
		filename = ls[i]
		hash = filename.replace('.json', '')
		# Load and standardize the ground truth.
		machine = load_json(os.path.join(args.machine_dir, filename))
		gt = load_json(os.path.join(args.gt_dir, filename))

		# Get segment timestamps.
		seg_ts = []
		seg_tag = []
		current_tag = None
		for j, tag in enumerate(gt['speaker_tags']):
			# If first element, start the sequence OR
			# If we encounter a different tag, start a new sequence.
			if j == 0 or tag != current_tag:
				current_tag = tag
				seg_ts.append(gt['timestamps'][j])
				seg_tag.append(tag)

		# Create segments for GT and machine using timestamps.
		machine_segments = create_segments(seg_ts, machine)
		gt_segments = create_segments(seg_ts, gt)

		# Compute sentence-level metrics.
		for j in range(len(gt_segments)):
			reference = gt_segments[j].split(' ')
			hypothesis = machine_segments[j].split(' ')
			bleu = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
			wer = word_error_rate(hypothesis, reference)
			gleu = evaluation.gleu.sentence_gleu([reference], hypothesis)
			# Write to file.
			result = f'{hash},{seg_ts[j]},{seg_tag[j]},{bleu},{gleu},{wer}'
			results_file.write(result + '\n')
			gt_out_file.write(gt_segments[j] + '\n')
			# Bug where machine_segments[j] is duplicating itself.
			stop_idx = int(len(machine_segments[j]) / 2)
			machine_out_file.write(machine_segments[j][:stop_idx] + '\n')

	results_file.close()
	gt_out_file.close()
	machine_out_file.close()


def create_segments(seg_ts: List[float], data: Dict) -> List[str]:
	"""
	Takes a transcription dictionary and returns a list of segments.
	Each segment are the words occurring between two timestamps.

	:param seg_ts: List of timestamps denoting the start time of the segment.
	:param data: Dictionary with keys: timestamps, speakerTags, words.
	:return: List of segments, where each segment is a string.
	"""
	segments: List[str] = []

	data_ts = np.asarray(data['timestamps'])
	# Process everything except the last segment/bucket.
	for i in range(0, len(seg_ts) - 1):
		start = seg_ts[i]
		end = seg_ts[i+1]
		# Find all words with this segment's timestamps.
		idx = np.where(np.logical_and(start <= data_ts, data_ts < end))[0]
		# Grab the actual words.
		if len(idx) > 0:
			buffer = []
			for j in idx:
				buffer.append(data['words'][j])
			segments.append(' '.join(buffer))
		# If we have no words, append an empty segment.
		else:
			segments.append('')

	# Process the last segment/bucket.
	idx = np.where(seg_ts[-1] <= data_ts)[0]
	if len(idx) == 0:
		segments.append('')
	else:
		buffer = []
		for j in idx:
			buffer.append(data['words'][j])
		seg = ' '.join(buffer)
		segments.append(seg)

	segments = [preproc.util.canonicalize_sentence(x) for x in segments]
	return segments


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
	speaker_tags = []
	# For each word, add it to our list.
	for B in A['results']:
		for C in B['alternatives']:
			for D in C['words']:
				# Get the core content.
				start_time = float(D['startTime'].replace('s', ''))
				timestamps.append(start_time)
				word = D['word']
				word = preproc.util.canonicalize_word(word)
				words.append(word)

				# Add the speaker information.
				if 'speakerTag' in D:
					speaker_tags.append(D['speakerTag'])
				else:
					speaker_tags.append('')

	result = {'timestamps': timestamps, 'words': words, 'speaker_tags': speaker_tags}
	return result


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

	d = Levenshtein.distance(''.join(w1), ''.join(w2))
	wer = d / max(len(target), 1)
	wer = min(wer, 1.0)
	return wer


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('machine_dir', type=str, help='Location of the predicted json transcriptions.')
	parser.add_argument('gt_dir', type=str, help='Location of the ground truth json transcription.')
	main(parser.parse_args())
