"""
Checks if a specific word exists in ANY of
the ground truth or machine results.
"""
import os
import sys
import json
import argparse
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Any, Tuple
import preproc.util


def main(args):
	if not os.path.exists(args.machine_dir):
		print(f'Path does not exist: {args.machine_dir}')
		sys.exit(1)
	if not os.path.exists(args.gt_dir):
		print(f'Path does not exist: {args.gt_dir}')
		sys.exit(1)

	gt_counter = load_and_count_json(args.gt_dir)
	machine_counter = load_and_count_json(args.machine_dir)

	assert(args.percent > 0)
	result = find_untranscribed_words(gt_counter, machine_counter, args.percent)

	return 0


def find_untranscribed_words(gt: Counter, machine: Counter, percent: int) -> List[Dict[str, any]]:
	"""
	Finds untranscribed words.

	That is, we find if there exist words in the GT which never occur in the machine transcription.

	:param gt: Counter of GT words.
	:param machine: Counter of machine words.
	:param percent: Flag the word if the machine reports less than (`percent`)% than reported in GT.
	:return: List of word/counts which occur in GT but not (or infrequently) in the machine transcription.
	"""
	result: List[Dict[str, any]] = []
	for word, gt_count in gt.most_common():
		if word not in machine:
			machine_count = 0
		else:
			machine_count = machine[word]
		if machine_count < gt_count * percent:
			r = {'word': word, 'machine': machine_count, 'gt': gt_count}
			result.append(r)
	return result


def load_and_count_json(fqn: str) -> Counter:
	"""
	Loads a json file and counts the occurrence of words.
	:param fqn: Path to the json file.
	:return: Counter of words.
	"""
	ls = os.listdir(fqn)
	counter = Counter()
	for i in tqdm(range(len(ls))):
		filename = ls[i]
		fqn = os.path.join(fqn, filename)
		with open(fqn, 'r') as f:
			A = json.load(f)

		for B in A['results']:
			for C in B['alternatives']:
				for D in C['words']:
					word = D['word']
					word = preproc.util.canonicalize(word)
					counter[word] += 1
	return counter


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('machine_dir', type=str,
						help='Location of the predicted json transcriptions.')
	parser.add_argument('gt_dir', type=str,
						help='Location of the ground truth json transcription.')
	parser.add_argument('--percent', default=0.1, type=float,
						help='Report words where machine is this % of the GT word count.')
	main(parser.parse_args())

