"""
This file takes the raw JSON output from the GT and Google API and
creates a nice paired-sentence JSON file which contains both the ground
truth and transcribed phrases, as well as some metadata.

This standardized format is used for all downstream metrics computation.

Example entry:
 '556': {
	 'hash': '0dd01803e57a3b95fcfab584bfb09aa604d80c0452ce5cd90f02669b0f9b9b5e',
	 'ts': 102.0,
	 'speaker': 'P',
	 'gt': 'hello there how are you doing',
	 'pred': 'hello their how arent you doing'
	}
"""
import re
import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import preproc.util
import evaluation.config as config


def main(args):
	if not os.path.exists(args.machine_dir):
		print(f'Path does not exist: {args.machine_dir}')
		sys.exit(0)
	if not os.path.exists(args.gt_dir):
		print(f'Path does not exist: {args.gt_dir}')
		sys.exit(0)

	# Creates a single dictionary, indexed by GID, and contains values: gt sentence, pred sentence, etc.
	paired: Dict[int, Dict] = {}

	# Loop over all sessions.
	ls = sorted(os.listdir(args.machine_dir))
	gid = 0  # Global sentence ID.
	for i in tqdm(range(len(ls)), desc='Processing Files'):
		filename = ls[i]
		hash_ = filename.replace('.json', '')
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

		# Create the final dictionary.
		for j in range(len(gt_segments)):
			value = {
				'hash': hash_,
				'ts': seg_ts[j],
				'speaker': seg_tag[j],
				'gt': gt_segments[j],
				'pred': machine_segments[j],
			}
			paired[gid] = value
			gid += 1

	# Write output json file.
	with open(config.PAIRED_FQN, 'w') as f:
		json.dump(paired, f, indent=2)


def is_repeated(arr: List[str], repeat: int) -> bool:
	"""
	Checks whether a segment array of strings is doubled. That is,
	the first half contains the same elements as the second half.
	:param arr: List of strings.
	:param repeat: Check if `arr` is repeated `repeat` times.
	:return: True if array is doubled, False otherwise.
	"""
	if len(arr) % repeat != 0:
		return False

	pointers = np.arange(0, repeat) * int(len(arr) / repeat)

	while pointers[-1] < len(arr):
		unique = set([arr[idx] for idx in pointers])
		if len(unique) > 1:
			return False
		pointers += 1

	return True


def is_doubled(arr: List[str]) -> bool:
	"""
	Checks whether a segment array of strings is doubled. That is,
	the first half contains the same elements as the second half.
	:param arr: List of strings.
	:return: True if array is doubled, False otherwise.
	"""
	if len(arr) % 2 != 0:
		return False

	first = 0
	second = int(len(arr) / 2)
	while second < len(arr):
		if arr[first] != arr[second]:
			return False
		first += 1
		second += 1
	return True


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
			if is_repeated(buffer, 2):
				clean = buffer[:int(len(buffer)/2)]
			elif is_repeated(buffer, 3):
				clean = buffer[:int(len(buffer)/3)]
			else:
				clean = buffer
			sentence = ' '.join(clean)
			sentence = re.sub('\s+', ' ', sentence).strip()
			segments.append(sentence)
		# If we have no words, append an empty segment.
		else:
			segments.append('')

	# Ignore the last segment because we don't know when it ends.
	# That is, the ground truth can end at 20 minutes, but the audio goes until 30 minutes. As a result,
	# we will have 10 minutes of machine transcript that will be considered wrong.
	# Process the last segment/bucket.
	# idx = np.where(seg_ts[-1] <= data_ts)[0]
	# if len(idx) == 0:
	# 	segments.append('')
	# else:
	# 	buffer = []
	# 	for j in idx:
	# 		buffer.append(data['words'][j])
	# 	seg = ' '.join(buffer)
	# 	segments.append(seg)

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
			# Sometimes the 'words' key is not present.
			if 'words' not in C:
				continue
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('machine_dir', type=str, help='Location of the predicted json transcriptions.')
	parser.add_argument('gt_dir', type=str, help='Location of the ground truth json transcription.')
	main(parser.parse_args())
