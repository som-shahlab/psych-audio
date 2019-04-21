"""
Extracts sentence-level word2vec features via mean pooling.
"""
import os
import sys
import argparse
import numpy as np
import gensim.models
from tqdm import tqdm
import evaluation.util
from typing import Dict, List, Optional, Callable

# Path to Google's pre-trained word2vec model.
MODEL_FQN = '/vol0/psych_audio/ahaque/models/word2vec/GoogleNews-vectors-negative300.bin'

# Dimension of word2vec.
F = 300


def main(args):
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	all_sentences, all_gids = evaluation.util.load_gt_pred_text_file(skip_empty=True)

	if args.model_name == 'word2vec':
		print('Creating Word2Vec model...')
		model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_FQN, binary=True)
		encode_fn: Callable = sentence_word2vec

	# Create the output accumulators.
	sentences = {'gt': [], 'pred': []}
	embeddings = {'gt': [], 'pred': []}
	gids = {'gt': [], 'pred': []}

	# Compute the embeddings.
	for i in tqdm(range(len(all_sentences['gt']))):
		# Compute embeddings for GT and pred.
		embed_gt = encode_fn(model, all_sentences['gt'][i])
		embed_pred = encode_fn(model, all_sentences['pred'][i])

		# Embedding might be None if entire sentence is out-of-vocab.
		if embed_gt is None or embed_pred is None:
			continue

		embeddings['gt'].append(embed_gt)
		embeddings['pred'].append(embed_pred)
		gids['gt'].append(int(all_gids['gt'][i]))
		gids['pred'].append(int(all_gids['pred'][i]))

	# Convert to numpy.
	pred_sentences = np.asarray(sentences['pred'])
	gt_sentences = np.asarray(sentences['gt'])
	pred_embeddings = np.asarray(embeddings['pred'])
	gt_embeddings = np.asarray(embeddings['gt'])
	pred_gids = np.asarray(gids['pred'])
	gt_gids = np.asarray(gids['gt'])

	# Save the final npz files.
	pred_fqn = os.path.join(args.output_dir, f'{args.model_name}_pred.npz')
	gt_fqn = os.path.join(args.output_dir, f'{args.model_name}_gt.npz')
	np.savez_compressed(pred_fqn, embeddings=pred_embeddings, gids=pred_gids, text=pred_sentences)
	np.savez_compressed(gt_fqn, embeddings=gt_embeddings, gids=gt_gids, text=gt_sentences)
	print(pred_fqn)
	print(gt_fqn)


def sentence_word2vec(model, sentence: str) -> Optional[np.ndarray]:
	"""
	Encodes a sentence into a word2vec sentence embedding.
	:param model: Word2vec or BERT model.
	:param sentence: Sentence as a string.
	:return embedding: Numpy array of the embedding.
	"""
	words = sentence.split(' ')
	# Count the number of non-zero word2vecs.
	count = 0
	for i, word in enumerate(words):
		if word in model.vocab:
			count += 1
	if count == 0:
		return None
	embeddings = np.zeros((count, F), np.float32)
	# Compute embeddings for each word.
	idx = 0
	for word in words:
		if word in model.vocab:
			embeddings[idx] = model[word]
			idx += 1
	# Mean pooling.
	embedding = embeddings.mean(0)
	return embedding


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model_name', type=str, choices=['word2vec', 'glove'])
	parser.add_argument('output_dir', type=str, help='Location to save the output npz files.')
	main(parser.parse_args())
