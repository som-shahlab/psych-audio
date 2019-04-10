"""
This file contains functions for evaluating the Google Speech API transcription performance.
"""
import os
import sys
import json
import argparse


def main(args):
	if not os.path.exists(args.machine_dir):
		print(f'Path does not exist: {args.machine_dir}')
		sys.exit(0)
	if not os.path.exists(args.gt_dir):
		print(f'Path does not exist: {args.gt_dir}')
		sys.exit(0)

	for filename in os.listdir(args.machine_dir):
		# Load and standardize the ground truth.
		machine = load_json(os.path.join(args.machine_dir, filename))

	# Load and standardize the machine output.
	# gt = load_gt(os.path.join(args.gt_dir, filename))
	# compute_stats(machine, gt)


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
	# For each word, add it to our list.
	for B in A['results']:
		print(B)
		for C in B['alternatives']:
			for D in C['words']:
				timestamps.append(D['startTime'])
				words.append(D['word'])
	return timestamps, words


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('machine_dir', type=str, help='Location of the predicted json transcriptions.')
	parser.add_argument('gt_dir', type=str, help='Location of the ground truth json transcription.')
	main(parser.parse_args())
