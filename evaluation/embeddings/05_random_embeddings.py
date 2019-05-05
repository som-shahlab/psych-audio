"""
This script generates random sentences and analyzes the embedding distances.

This script must be run on a linux system.
"""
import os
import sys
import argparse
import scipy.stats
import numpy as np
from typing import *
import scipy.spatial.distance
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import evaluation.util
from evaluation.embeddings import config
import evaluation.embeddings.util

# Location to save and load the random embedding npy files.
embedding_dir = '/home/ahaque/Desktop/random/'


# Location of the random words file. This file contains 1 word per line.
word_file = '/usr/share/dict/words'


def main(args):
	# Create the output dir.
	if not os.path.exists(embedding_dir):
		os.makedirs(embedding_dir)
	
	# Auto-detect if we need to create embeddings.
	ls = os.listdir(embedding_dir)
	fqn = ''
	has_file = False
	if len(ls) == 0:
		generate_embeddings(args)
	else:
		# Make sure we have a npy for this embedding.
		for filename in ls:
			if args.embedding_name == filename[:-4]:
				fqn = os.path.join(embedding_dir, filename)
				has_file = True
	
	# Compute distances.
	if has_file:
		print(f'Loading: {fqn}')
		A = np.load(fqn)[:200]
		cosine = flat_pairwise(A, 'cosine')
		euclidean = flat_pairwise(A, 'euclidean')
		wasserstein = flat_wasserstein(A)

		# Create the subpanels.
		fig, axes = plt.subplots(1, 2, figsize=(15, 6))
		axes[0].hist(cosine, 100, density=True, facecolor='g', alpha=0.75, label='cosine')
		axes[0].hist(euclidean, 100, density=True, facecolor='b', alpha=0.75, label='euclidean')
		axes[0].hist(wasserstein, 100, density=True, facecolor='r', alpha=0.75, label='wasserstein')
		axes[0].set_xlabel('Distance')
		axes[0].set_ylabel('Frequency')
		axes[0].set_ylim([0, 5])
		axes[0].legend()

		# T-SNE plot.
		print('Fitting T-SNE...')
		model = TSNE(n_components=2)
		X = model.fit_transform(A)
		x, y = X[:, 0], X[:, 1]
		axes[1].scatter(x, y, label='Sentence')
		axes[1].legend()
		axes[1].set_title('T-SNE of Embeddings')
		#plt.scatter(x, y)
		plt.show()


def flat_wasserstein(A: np.ndarray):
	"""
	Computes the pairwise wasserstein distance.
	
	Args:
		A (np.ndarray): (N, embedding_size) matrix.
	"""
	# Create the parallel arrays for comparison.
	N = int(((len(A) ** 2) - len(A)) / 2)

	flat = np.zeros((N,), np.float32)
	idx = 0
	for i in range(len(A)):
		for j in range(i + 1, len(A)):
			flat[idx] = scipy.stats.wasserstein_distance(A[i], A[j])
			idx += 1

	print(f'------ wasserstein ------')
	print(f'Mean: {flat.mean():.4f}')
	print(f'Std: {flat.std():.4f}')
	print(f'n: {len(flat)}')
	return flat


def flat_pairwise(A: np.ndarray, metric: str):
	"""Computes the pairwise distance and returns a vector of distances."""
	dists = scipy.spatial.distance.cdist(A, A, metric=metric)
	idx1, idx2 = np.nonzero(np.tril(dists))  # Only time dist=0 is if exact same sentence, which we want to discard.
	flat = dists[idx1, idx2]
	print(f'------ {metric} ------')
	print(f'Mean: {flat.mean():.4f}')
	print(f'Std: {flat.std():.4f}')
	print(f'n: {len(flat)}')
	return flat


def generate_embeddings(args):
	"""
	Generates random sentences and embeddings and stores them in `embedding_dir`.
	
	Args:
		args: Argparse namespace.
	"""
	# Load the embedding model.
	print(f'Loading the {args.embedding_name} model...')
	model, keys = evaluation.embeddings.util.load_embedding_model(args.embedding_name)

	# Load the vocabulary.
	with open(word_file, 'r') as f:
		vocab = [x.strip().lower() for x in f.readlines()]

	# Generate N sentences.
	sentences = []
	for _ in range(args.n):
		s = generate_sentence(vocab)
		sentences.append(s)

	embeddings = np.zeros((args.n, config.F[args.embedding_name]), np.float32)
	for i, s in enumerate(sentences):
		embed = evaluation.embeddings.util.encode_from_dict(args.embedding_name, model, keys, s)
		if embed is None:
			print(f'No embedding: {s}')
		else:
			embeddings[i] = embed

	fqn = os.path.join(embedding_dir, f'{args.embedding_name}.npy')
	np.save(fqn, embeddings)
	print(fqn)

	

def generate_sentence(vocab: List[str]) -> str:
	"""
	Generates a random English sentence.
	:return sentence: Generated sentence.
	"""
	# Determine sentence length.
	n_words = int(np.clip(np.random.normal(8, 2), 0, 15))

	# Select words.
	vocab_size = len(vocab)
	idx = np.random.choice(np.arange(0, vocab_size), (n_words,), replace=True)

	# Compose the sentence.
	sentence = ''
	for i in idx:
		sentence += f' {vocab[i]}'

	sentence = evaluation.util.canonicalize(sentence)
	return sentence


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('n', type=int, default=100, help='Number of sentences to generate.')
	parser.add_argument('embedding_name', type=str, choices=['word2vec', 'glove'])
	main(parser.parse_args())
