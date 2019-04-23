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


# Path to Google's pre-trained word2vec model (.bin file)
WORD2VEC_MODEL_FQN: str = '/vol0/psych_audio/ahaque/models/word2vec/GoogleNews-vectors-negative300.bin'

# Path to Stanford's pre-trained GloVe model (.txt file)
GLOVE_MODEL_FQN: str = '/vol0/psych_audio/ahaque/models/glove/glove.840B.300d.txt'

# Dimension of each embedding.
F: Dict[str, int] = {'word2vec': 300, 'glove': 300, 'bert': 1024}


def main(args):
	# Check if the output dir exists.
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Load the sentences.
	paired = evaluation.util.load_paired_json(skip_empty=True)

	# Select and load the embedding model.
	print(f'Loading the {args.embedding_name} model...')
	if args.embedding_name == 'word2vec':
		model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_FQN, binary=True)
		keys = model.vocab
		encode_fn: Callable = encode_from_dict
	elif args.embedding_name == 'glove':
		model = load_glove()
		keys = set(model.keys())
		encode_fn: Callable = encode_from_dict

	# Create the output accumulators.
	sentences = {'gt': [], 'pred': []}
	embeddings = {'gt': [], 'pred': []}
	out_gids = {'gt': [], 'pred': []}

	# Compute the embeddings.
	paired_gids = list(paired.keys())
	for i in tqdm(range(len(paired_gids)), desc='Embedding'):
		gid = paired_gids[i]
		# Compute embeddings for GT and pred.
		embed_gt = encode_fn(args, model, keys, paired[gid]['gt'])
		embed_pred = encode_fn(args, model, keys, paired[gid]['pred'])

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
	pred_fqn = os.path.join(args.output_dir, f'{args.embedding_name}_pred.npz')
	gt_fqn = os.path.join(args.output_dir, f'{args.embedding_name}_gt.npz')
	np.savez_compressed(pred_fqn, embeddings=pred_embeddings, gids=pred_gids, text=pred_sentences)
	np.savez_compressed(gt_fqn, embeddings=gt_embeddings, gids=gt_gids, text=gt_sentences)

	print(pred_fqn)
	print(gt_fqn)


def encode_from_dict(args, model, keys: Set[str], sentence: str) -> Optional[np.ndarray]:
	"""
	Encodes a sentence using a dictionary-based embedding. That is, either word2vec or Glove.
	A dictionary-based embedding has words as the keys and a numpy array as the value.

	:param args: Argparse namespace.
	:param model: Glove or word2vec model (usually a dictionary-like structure).
	:param keys: Set of valid words in the model.
	:param sentence: Sentence as a string.
	:return embedding: Numpy array of the sentence embedding.
	"""
	words = sentence.split(' ')
	# Count the number of words for which we have an embedding.
	count = 0
	for i, word in enumerate(words):
		if word in keys:
			count += 1
	if count == 0:
		return None

	# Get embeddings for each word.
	embeddings = np.zeros((count, F[args.embedding_name]), np.float32)
	idx = 0
	for word in words:
		if word in keys:
			embeddings[idx] = model[word]
			idx += 1

	# Mean pooling.
	embedding = embeddings.mean(0)
	return embedding


def load_glove() -> Dict[str, np.ndarray]:
	"""Loads the GloVe model."""
	model: Dict[str, np.ndarray] = {}
	with open(GLOVE_MODEL_FQN, 'r') as f:
		for line in f:
			tokens = line.strip().split(' ')
			word = tokens[0]
			vec = np.array([float(x) for x in tokens[1:]])
			model[word] = vec
	return model


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('embedding_name', type=str, choices=['word2vec', 'glove'])
	parser.add_argument('output_dir', type=str, help='Location to save the output npz files.')
	main(parser.parse_args())
