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

# Location of the saved embeddings (npz files).
NPZ_DIR = '/vol0/psych_audio/ahaque/psych-audio/results/embeddings'

# Where to save the output csv files.
OUT_DIR = '/vol0/psych_audio/ahaque/psych-audio/results/dists'


def main(args):
	# Check if the files exist.
	gt_fqn = os.path.join(NPZ_DIR, f'{args.embedding_name}_gt.npz')
	pred_fqn = os.path.join(NPZ_DIR, f'{args.embedding_name}_pred.npz')
	if not os.path.exists(gt_fqn):
		print(f'Does not exist: {gt_fqn}')
		sys.exit(1)
	if not os.path.exists(pred_fqn):
		print(f'Does not exist: {pred_fqn}')
		sys.exit(1)

	# Load the files.
	print(f'Loading {args.embedding_name} npz files...')
	gt = np.load(gt_fqn)
	gt_embeddings, gt_gids = gt['embeddings'], gt['gids']
	pred = np.load(pred_fqn)
	pred_embeddings = pred['embeddings']

	out_fqn = os.path.join(OUT_DIR, f'{args.embedding_name}.csv')
	with open(out_fqn, 'w') as f:
		for i in tqdm(range(len(gt['gids']))):
			gid = gt_gids[i]
			a = gt_embeddings[i]
			b = pred_embeddings[i]
			# For BERT, need to convert from word-level to sentence-level.
			if args.embedding_name == 'bert':
				# a: (max_seq_len, 1024)
				# If a row is all zeros, it means BERT did not use that row. Therefore we can skip it.
				a_idx = np.sum(a, 1) > 0
				b_idx = np.sum(b, 1) > 0

				# If either a or b contains empty embeddings, skip.
				if np.sum(a_idx) == 0 or np.sum(b_idx) == 0:
					continue
				else:
					a = a[np.sum(a, 1) > 0].mean(0)
					b = b[np.sum(b, 1) > 0].mean(0)

			# Compute the distance.
			dist = scipy.spatial.distance.cosine(a, b)
			f.write(f'{gid},{dist}\n')

	print(out_fqn)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('embedding_name', type=str, choices=['word2vec', 'glove', 'bert'])
	main(parser.parse_args())
