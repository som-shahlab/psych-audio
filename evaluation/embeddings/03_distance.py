"""
Computes various distance-based metrics between the GT and predicted embedding vectors.

Prerequisites:
- Glove and/or Word2Vec embeddings computed for each GT and predicted sentence. You should have npz files.
- BERT embeddings computed. This should also be a npz file.

Outputs:
- One CSV file per embedding. Each line of the CSV file contains GID,cosine_distance
"""
import os
import sys
import argparse
import numpy as np
import scipy.spatial.distance
from tqdm import tqdm


def main(args):
	# Check if the files exist.
	gt_fqn = os.path.join(args.npz_dir, f'{args.embedding_name}_gt.npz')
	pred_fqn = os.path.join(args.npz_dir, f'{args.embedding_name}_pred.npz')
	if not os.path.exists(gt_fqn):
		print(f'Does not exist: {gt_fqn}')
		sys.exit(1)
	if not os.path.exists(pred_fqn):
		print(f'Does not exist: {pred_fqn}')
		sys.exit(1)

	# Load the files.
	print(f'Loading {args.embedding_name} npz files...')
	gt = np.load(gt_fqn)
	pred = np.load(pred_fqn)

	out_fqn = os.path.join(args.out_dir, f'{args.embedding_name}.csv')
	with open(out_fqn, 'w') as f:
		for i in tqdm(range(len(gt['gids']))):
			gid = gt['gids'][i]
			a = gt['embeddings'][i]
			b = pred['embeddings'][i]
			dist = worker(args.embedding_name, gid, a, b)
			f.write(f'{gid},{dist}\n')


def worker(name: str, gid: int, a: np.ndarray, b: np.ndarray) -> float:
	"""
	Computes the distance between two vectors and places the result in a queue.
	:param name: Name of the embedding.
	:param gid: Global ID.
	:param a: First vector (or tensor for BERT).
	:param b: Second vector (or tensor for BERT).
	"""
	# For BERT, need to convert from word-level to sentence-level.
	if name == 'bert':
		# a: (max_seq_len, 1024)
		# If a row is all zeros, it means BERT did not use that row. Therefore we can skip it.
		a = a[np.sum(a, 1) > 0].mean(0)
		b = b[np.sum(b, 1) > 0].mean(0)

	# Compute the distance.
	dist = scipy.spatial.distance.cosine(a, b)
	return dist


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('embedding_name', type=str, choices=['word2vec', 'glove', 'bert'])
	parser.add_argument('npz_dir', type=str, help='Location of the embedding npz files.')
	parser.add_argument('out_dir', type=str, help='Where to place the final csv files.')
	main(parser.parse_args())
