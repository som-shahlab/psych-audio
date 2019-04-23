"""
This file creates a Table 2 (quant results) but for the different embeddings.

Prerequisites:
- Embeddings saved as npz files in `results/`.
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import scipy.spatial.distance
from typing import List, Dict, Set


def main(args):
	if not os.path.exists(args.data_dir):
		print(f'Does not exist: {args.data_dir}')
		sys.exit(1)

	# Load and compute the embeddings for each sentence.
	names = autodetect_names(args.data_dir)

	# For each embedding type, compute the cosine distance and save to json.
	for name in names:
		# Load the npz files.
		gt = np.load(os.path.join(args.data_dir, f'{name}_gt.npz'))
		pred = np.load(os.path.join(args.data_dir, f'{name}_pred.npz'))

		# Compute the distances.
		for i in tqdm(range(len(gt['gids']))):
			# BERT has word-level embeddings, so we need to collapse it into a sentence-level embedding.
			if name == 'bert':
				raise NotImplementedError
			else:
				a = gt['embeddings'][i]
				b = pred['embeddings'][i]
			# Compute distance.
			gid = gt['gids'][i]
			dist = scipy.spatial.distance.cosine(a, b)


def autodetect_names(data_dir: str) -> List[str]:
	"""
	Automatically detects the embedding names. e.g. bert, word2vec
	:param data_dir: Path to the npz files.
	:return names: List of embedding names.
	"""
	filenames: List[str] = os.listdir(data_dir)
	names: Set[str] = set()
	for filename in filenames:
		key1 = '_gt.npz'
		key2 = '_pred.npz'
		if key1 in filename:
			idx = filename.index(key1)
		else:
			idx = filename.index(key2)
		name = filename[:idx]
		names.add(name)
	names: List[str] = list(names)
	return names


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_dir', type=str, help='Location of the embedding npz files.')
	parser.add_argument('out_dir', type=str, help='Where to save the output csv files containing cosine distances.')
	main(parser.parse_args())
