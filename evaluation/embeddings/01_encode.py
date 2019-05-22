"""
Extracts sentence-level features via mean-pooling.
This script supports both the word2vec and glove models.
Run this script before computing cosine distance.

Estimated runtimes on d01:
- Word2Vec: 2 minutes
- Glove: 4 minutes
"""
import os
import argparse
import numpy as np
import gensim.models
from typing import *
from tqdm import tqdm
import evaluation.util
import evaluation.embeddings.util
from evaluation import config


def main(args):
	# Check if the output dir exists.
	if not os.path.exists(config.NPZ_DIR):
		os.makedirs(config.NPZ_DIR)

	# Load the sentences.
	paired = evaluation.util.load_paired_json(skip_empty=True)

	# Select and load the embedding model.
	print(f'Loading the {args.embedding_name} model...')
	model, keys = evaluation.embeddings.util.load_embedding_model(args.embedding_name)

	# Create the output accumulators.
	sentences = {'gt': [], 'pred': []}
	embeddings = {'gt': [], 'pred': []}
	out_gids = {'gt': [], 'pred': []}

	# Compute the embeddings.
	paired_gids = list(paired.keys())
	for i in tqdm(range(len(paired_gids)), desc='Embedding'):
		gid = paired_gids[i]
		# Compute embeddings for GT and pred.
		embed_gt = evaluation.embeddings.util.encode_from_dict(args.embedding_name, model, keys, paired[gid]['gt'])
		embed_pred = evaluation.embeddings.util.encode_from_dict(args.embedding_name, model, keys, paired[gid]['pred'])

		# Embedding might be None if entire sentence is out-of-vocab.
		if embed_gt is None or embed_pred is None:
			continue

		# Append to list.
		embeddings['gt'].append(embed_gt)
		embeddings['pred'].append(embed_pred)
		out_gids['gt'].append(int(gid))
		out_gids['pred'].append(int(gid))

	# Convert to numpy.
	pred_sentences, gt_sentences = np.asarray(sentences['pred']), np.asarray(sentences['gt'])
	pred_embeddings, gt_embeddings = np.asarray(embeddings['pred']), np.asarray(embeddings['gt'])
	pred_gids, gt_gids = np.asarray(out_gids['pred']), np.asarray(out_gids['gt'])

	# Save the final npz files.
	pred_fqn = os.path.join(config.NPZ_DIR, f'{args.embedding_name}_pred.npz')
	gt_fqn = os.path.join(config.NPZ_DIR, f'{args.embedding_name}_gt.npz')
	np.savez_compressed(pred_fqn, embeddings=pred_embeddings, gids=pred_gids, text=pred_sentences)
	np.savez_compressed(gt_fqn, embeddings=gt_embeddings, gids=gt_gids, text=gt_sentences)

	print(pred_fqn)
	print(gt_fqn)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('embedding_name', type=str, choices=['word2vec', 'glove'])
	main(parser.parse_args())
