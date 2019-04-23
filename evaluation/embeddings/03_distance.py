"""
Computes various distance-based metrics between the GT and predicted embedding vectors.

Prerequisites:
- Glove and/or Word2Vec embeddings computed for each GT and predicted sentence. You should have npz files.
- BERT embeddings computed. This should also be a npz file.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.spatial.distance
import evaluation.util


def main(args):
	gt = np.load('bert_gts.npz')
	pred = np.load('bert_preds.npz')

	# For each GID, find whether it was speaker or therapist.
	gid2speaker = evaluation.util.load_gid2speaker()
	gt_gids = np.asarray([int(x) for x in gt['gids']])  # Convert from string to int, since gids are ints.
	T_dists = []
	P_dists = []
	print('Computing distances...')
	max_gid = int(gt_gids.max())

	for gid in range(max_gid):
		# If we do not have embeddings for this gid, it means
		# either GT or pred was empty string. Therefore the embedding distance = 1.
		speaker = gid2speaker[gid]

		if gid not in gt_gids:
			dist = 1
		else:
			idx = np.where(gt_gids == gid)[0]
			gt_embedding = gt['bert'][idx]
			pred_embedding = pred['bert'][idx]
			dist = scipy.spatial.distance.cosine(gt_embedding, pred_embedding)

		if speaker == 'P':
			P_dists.append(dist)
		elif speaker == 'T':
			T_dists.append(dist)

	# Compute mean.
	T_dists = np.asarray(T_dists)
	P_dists = np.asarray(P_dists)

	print('-' * 40)
	print('Phrase-Level')
	print(f'T (n={len(T_dists)})', T_dists.mean())
	print(f'P (n={len(P_dists)})', P_dists.mean())


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	main(parser.parse_args())
