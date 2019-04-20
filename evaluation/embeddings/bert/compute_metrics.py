"""Computes metrics after BERT embeddings have been extracted."""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.spatial.distance

PHRASE_RESULTS = '/vol0/psych_audio/ahaque/psych-audio/results/phrase.csv'


def main(args):
	gt = np.load('bert_gts.npz')
	pred = np.load('bert_preds.npz')

	# For each GID, find whether it was speaker or therapist.
	df = pd.read_csv(PHRASE_RESULTS, sep=',')
	gid2speaker = []
	print('Building gid2speaker map...')
	for _, row in df.iterrows():
		gid2speaker.append(row['tag'])

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
