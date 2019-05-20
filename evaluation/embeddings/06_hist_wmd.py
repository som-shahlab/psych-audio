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
	model = api.load('word2vec-google-news-300')

	print('Computing random dists...')
	random_dists = pairwise_wmd(model, random_sentences)
	eeu.print_metrics(random_dists, 'WMD-random')

	print(f'Computing corpus dists...')
	corpus_dists = pairwise_wmd(model, corpus_sentences)
	eeu.print_metrics(corpus_dists, 'WMD-corpus')

	print('--------- Unequal Variance --------')
	statistic, pval = scipy.stats.ttest_ind(random_dists, corpus_dists, equal_var=False)
	print(f'Statistic: {statistic}')
	print(f'Two-Tailed P Value: {pval}')
	print(f'n sentence: {args.n}')
	print(f'n corpus: {len(corpus_dists)}')
	print(f'n random: {len(random_dists)}')


def pairwise_wmd(model, sentences: List[str]) -> np.ndarray:
	"""
	Computes pairwise Word Mover Distance (WMD) for a list of strings.
	
	Args:
		model: Gensim model.
		sentences (List[str]): List of sentences.
	
	Returns:
		dists: Numpy array of WMD distances.
	"""
	# WMD requires each sentence as a List of words.
	sentences = [x.split() for x in sentences]
	dists: List[float] = []
	n = len(sentences)
	n_dists = int((n ** 2 - n) / 2)
	pbar = tqdm(total=n_dists)
	for i in range(n):
		for j in range(i + 1, n):
			pbar.update(1)
			d = model.wmdistance(sentences[i], sentences[j])
			if math.isnan(d) or d <= 0 or math.isinf(d):
				continue
			dists.append(d)
	pbar.close()
	dists = np.asarray(dists)
	return dists


if __name__ == '__main__':
	plt.style.use('ggplot')
	parser = argparse.ArgumentParser()
	parser.add_argument('output_dir', help='Where to save the output figure.')
	parser.add_argument('--n', type=int, help='Number of sentences to use for pairwise comparison.')
	main(parser.parse_args())
