"""
Creates the histogram figure which shows the similarity/difference
between randomly generated sentences and sentences from our corpus.
"""
import os
import sys
import math
import argparse
import numpy as np
from typing import *
import scipy.stats
from tqdm import tqdm
import sklearn.preprocessing
import scipy.spatial.distance
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gensim.downloader as api


import evaluation.embeddings.util as eeu
from evaluation.embeddings import config

def main(args):
	# Generate random sentences.
	random_sentences: List[str] = eeu.random_sentences(args.n, use_corpus=False)

	# Select N sentences from our GT corpus.
	corpus_sentences: List[str] = eeu.random_sentences(args.n, use_corpus=True)

	print(f'Loading the model..')
	model, keys = eeu.load_embedding_model('word2vec')

	print(f'Encoding...')
	random_embeddings = batch_encode(model, keys, random_sentences)
	corpus_embeddings = batch_encode(model, keys, corpus_sentences)

	print(corpus_embeddings)


def batch_encode(model: Dict[str, np.ndarray], keys: Set[str], sentences: List[str]) -> np.ndarray:
	"""
	Encodes a List of sentences into multiple embeddings.
	
	Args:
		model (Dict[str, np.ndarray]): KeyedVector word2vec model.
		keys (Set[str]): Set of words in the model's vocabulary.
		sentences (List[str]): List of sentences.
	
	Returns:
		np.ndarray: (N, F) embedding matrix.
	"""
	n = len(sentences)
	embeddings = np.zeros((n, config.F['word2vec']))
	for i in range(n):
		embed = eeu.encode_from_dict('word2vec', model, keys, sentences[i])
		embeddings[i] = embed
	return embeddings


def main2():

	print('Computing random dists...')
	random_dists = pairwise_wmd(model, random_sentences)
	print_metrics(random_dists, 'WMD-random')

	print(f'Computing corpus dists...')
	corpus_dists = pairwise_wmd(model, corpus_sentences)
	print_metrics(corpus_dists, 'WMD-corpus')

	n_comparisons = int((args.n ** 2 - args.n) / 2)
	print('--------- Unequal Variance --------')
	statistic, pval = scipy.stats.ttest_ind(random_dists, corpus_dists, equal_var=False)
	print(f'Statistic: {statistic}')
	print(f'Two-Tailed P Value: {pval}')
	print(f'n sentence: {args.n}')
	print(f'n corpus: {len(corpus_dists)}')
	print(f'n random: {len(random_dists)}')

	print('--------- Equal Variance --------')
	statistic, pval = scipy.stats.ttest_ind(random_dists, corpus_dists, equal_var=True)
	print(f'Statistic: {statistic}')
	print(f'Two-Tailed P Value: {pval}')
	print(f'n sentence: {args.n}')
	print(f'n corpus: {len(corpus_dists)}')
	print(f'n random: {len(random_dists)}')

	# For histogram to sum to 1, histogram requires integer=1 width bins. Our distances
	# are often very small. Therefore, we need to scale the distances (temporarily) to compute hist.
	n_bins = 30
	random_dists = (random_dists * n_bins).astype(np.int64)
	corpus_dists = (corpus_dists * n_bins).astype(np.int64)

	# Need to manually specify bins otherwise the histogram will not sum to 1.
	min_val = min(random_dists.min(), corpus_dists.min())
	max_val = max(random_dists.max(), corpus_dists.max())
	bins = np.arange(min_val, max_val)

	# Create the histogram.
	fig, axes = plt.subplots(1, 1, figsize=(16, 10))
	axes.hist(random_dists, bins=bins, density=True, facecolor='g', alpha=0.6, label='Random')
	axes.hist(corpus_dists, bins=bins, density=True, facecolor='b', alpha=0.6, label='Corpus')
	axes.set_xlabel('Distance')
	axes.set_ylabel('Probability')
	axes.legend()
	plot_fqn = os.path.join(args.output_dir, 'cosine.png')
	plt.savefig(plot_fqn)


def print_metrics(arr: np.ndarray, text: str):
	"""Prints various metrics for an array."""
	print(f'------ {text} ------')
	print(f'Mean: {arr.mean():.4f}')
	print(f'Std: {arr.std():.4f}')
	print(f'n: {len(arr)}')


if __name__ == '__main__':
	plt.style.use('ggplot')
	parser = argparse.ArgumentParser()
	parser.add_argument('output_dir', help='Where to save the output figure.')
	parser.add_argument('--n', type=int, help='Number of sentences to use for pairwise comparison.')
	main(parser.parse_args())
