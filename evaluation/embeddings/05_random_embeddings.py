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
from bert_serving.client import BertClient

import evaluation.util
from evaluation.embeddings import config
import evaluation.embeddings.util

# Location to save and load the random embedding npy files.
embedding_dir = '/home/ahaque/Desktop/random/'


def main(args):
	# Create the output dir.
	if not os.path.exists(embedding_dir):
		os.makedirs(embedding_dir)
	
	# Auto-detect if we need to create embeddings.
	ls = os.listdir(embedding_dir)
	fqn = ''
	has_file = False
	if len(ls) > 0:
		# Make sure we have a npy for this embedding.
		for filename in ls:
			if args.embedding_name == filename[:-4]:
				fqn = os.path.join(embedding_dir, filename)
				has_file = True

	if not has_file:
		generate_embeddings(args)
	
	# Compute distances.
	if has_file:
		print(f'Loading: {fqn}')
		A = np.load(fqn)
		cosine = flat_pairwise(A, 'cosine')
		euclidean = flat_pairwise(A, 'euclidean')
		wasserstein = flat_wasserstein(A)

		# Create the subpanels.
		fig, axes = plt.subplots(1, 2, figsize=(20, 9))
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
	print_metrics(flat, 'wasserstein')
	return flat


def flat_pairwise(A: np.ndarray, metric: str):
	"""Computes the pairwise distance and returns a vector of distances."""
	dists = scipy.spatial.distance.cdist(A, A, metric=metric)
	idx1, idx2 = np.nonzero(np.tril(dists))  # Only time dist=0 is if exact same sentence, which we want to discard.
	flat = dists[idx1, idx2]
	print_metrics(flat, metric)
	return flat


def print_metrics(arr: np.ndarray, text: str):
	"""Prints various metrics for an array."""
	print(f'------ {text} ------')
	print(f'Mean: {arr.mean():.4f}')
	print(f'Std: {arr.std():.4f}')
	print(f'n: {len(arr)}')


def generate_embeddings(args, bc=None):
	"""
	Generates random sentences and embeddings and stores them in `embedding_dir`.
	
	Args:
		args: Argparse namespace.
		bc: Bert Client.
	"""
	# Load the embedding model.
	if args.embedding_name == 'bert':
		print('Creating BERT client...')
		bc = BertClient(check_length=False)
		print('Done.')
	else:
		print(f'Loading the {args.embedding_name} model...')
		model, keys = evaluation.embeddings.util.load_embedding_model(args.embedding_name)

	# Generate N sentences.
	sentences = evaluation.embeddings.util.random_sentences(args.n)

	if args.embedding_name == 'bert':
		embeddings = bc.encode(sentences)

	if args.embedding_name in ['word2vec', 'glove']:
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


if __name__ == '__main__':
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size': 20})
	parser = argparse.ArgumentParser()
	parser.add_argument('n', type=int, default=100, help='Number of sentences to generate.')
	parser.add_argument('embedding_name', type=str, choices=['word2vec', 'glove', 'bert'])
	main(parser.parse_args())
