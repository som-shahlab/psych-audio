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


def main(args):
	if not os.path.exists(args.machine_dir):
		print(f'Path does not exist: {args.machine_dir}')
		sys.exit(0)
	if not os.path.exists(args.gt_dir):
		print(f'Path does not exist: {args.gt_dir}')
		sys.exit(0)

	results_file = open('results.csv', 'w')
	gt_out_file = open('gt_out.txt', 'w')
	machine_out_file = open('machine_out.txt', 'w')

	results_file.write('hash,speaker_ts,speaker_tag,bleu,wer\n')
	ls = os.listdir(args.machine_dir)
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
		gt_segments = create_segments(seg_ts, gt)
		machine_segments = create_segments(seg_ts, machine)

		# See: https://stackoverflow.com/questions/40542523/nltk-corpus-level-bleu-vs-sentence-level-bleu-score
		list_of_hypotheses = []
		list_of_references = []

		for j in range(len(gt_segments)):
			reference = gt_segments[j].split(' ')
			hypothesis = machine_segments[j].split(' ')
			references = [reference]
			bleu = nltk.translate.bleu_score.sentence_bleu(references=references, hypothesis=h)


		for ts in buckets:
			# Compose the reference and hypothesis.
			r = gt_phrases[ts]
			r_str = ' '.join(r)
			h = machine_phrases[ts]
			h_str = ' '.join(h)

			references = [r]
			list_of_hypotheses.append(h)
			list_of_references.append(references)

			# Compute metrics.
			bleu1 = nltk.translate.bleu_score.sentence_bleu(references=references, hypothesis=h)
			speakerTag = phrase_speakers[ts]

			# Write to file.
			word_error_rate = wer(h, r)
			result = f'{hash},{ts},{speakerTag},{bleu1},{word_error_rate}\n'
			results_file.write(result)
			gt_out_file.write(r_str + '\n')
			machine_out_file.write(h_str + '\n')

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
	buffer = []
	idx = 0  # Timestamp of the bucket we're currently composing.
	for i in range(len(data['timestamps'])):
		ts, word = data['timestamps'][i], data['words'][i]
		# Traverse the words/ts array and once we're past the current segment
		# timestamp, save the current seg and reset the buffer.
		if idx + 1 < len(seg_ts) and ts >= seg_ts[idx + 1]:
			sentence = ' '.join(buffer)
			sentence = preproc.util.canonicalize(sentence)
			segments.append(sentence)
			buffer = []
			idx += 1
		buffer.append(word)
	# Write the last buffer.
	sentence = ' '.join(buffer)
	sentence = preproc.util.canonicalize(sentence)
	segments.append(sentence)
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
				word = preproc.util.canonicalize(word)
				words.append(word)

				# Add the speaker information.
				if 'speakerTag' in D:
					speaker_tags.append(D['speakerTag'])
				else:
					speaker_tags.append('')

	result = {'timestamps': timestamps, 'words': words, 'speaker_tags': speaker_tags}
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
	wer = d / max(len(target), 1)
	wer = min(wer, 1.0)
	return wer


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('machine_dir', type=str, help='Location of the predicted json transcriptions.')
	parser.add_argument('gt_dir', type=str, help='Location of the ground truth json transcription.')
	main(parser.parse_args())
