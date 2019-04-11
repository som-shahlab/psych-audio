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


def main(args):
	if not os.path.exists(args.json_dir):
		print(f'Path does not exist: {args.json_dir}')
		sys.exit(0)

	ls = os.listdir(args.json_dir)
	counter = Counter()
	for i in tqdm(range(len(ls))):
		filename = ls[i]
		# Load and standardize the ground truth.
		fqn = os.path.join(args.json_dir, filename)
		with open(fqn, 'r') as f:
			A = json.load(f)

		for B in A['results']:
			for C in B['alternatives']:
				for D in C['words']:
					word = D['word']
					counter[word] += 1

	if args.word in counter:
		print(f'Found {counter[args.word]} of {args.word} in {args.json_dir}')
	else:
		print(f'{args.word} not found in {args.json_dir}')

	print('Most Common')
	print(counter.most_common(10))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('word', type=str, help='Word to search for.')
	parser.add_argument('json_dir', type=str, help='Location of the json files.')
	main(parser.parse_args())
