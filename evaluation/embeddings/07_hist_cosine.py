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
from evaluation import config

def main(args):
	# Generate random sentences.
	random_sentences: List[str] = eeu.random_sentences(args.n, use_corpus=False)

	# Select N sentences from our GT corpus.
	corpus_sentences: List[str] = eeu.random_sentences(args.n, use_corpus=True)

	print(f'Loading the model..')
	model, keys = eeu.load_embedding_model(args.embedding_name)

	print(f'Encoding...')
	random_embeddings = eeu.batch_encode(args.embedding_name, model, keys, random_sentences)
	corpus_embeddings = eeu.batch_encode(args.embedding_name, model, keys, corpus_sentences)

	print('Computing distance...')
	random_dists = eeu.pairwise_metric(random_embeddings, 'cosine')
	corpus_dists = eeu.pairwise_metric(corpus_embeddings, 'cosine')

	eeu.print_metrics(random_dists, 'Cosine-random')
	eeu.print_metrics(corpus_dists, 'Cosine-corpus')

	out_fqn = os.path.join(args.output_dir, 'cosine.png')
	eeu.plot_histogram(out_fqn, random_dists, corpus_dists, n_bins=50)

	print('--------- Unequal Variance --------')
	statistic, pval = scipy.stats.ttest_ind(random_dists, corpus_dists, equal_var=False)
	print(f'Statistic: {statistic}')
	print(f'Two-Tailed P Value: {pval}')
	print(f'n sentence: {args.n}')
	print(f'n corpus: {len(corpus_dists)}')
	print(f'n random: {len(random_dists)}')


if __name__ == '__main__':
	plt.style.use('ggplot')
	parser = argparse.ArgumentParser()
	parser.add_argument('output_dir', help='Where to save the output figure.')
	parser.add_argument('--embedding_name', default='word2vec', choices=['word2vec'])
	parser.add_argument('--n', type=int, help='Number of sentences to use for pairwise comparison.')
	main(parser.parse_args())
