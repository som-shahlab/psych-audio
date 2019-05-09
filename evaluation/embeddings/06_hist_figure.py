"""
Creates the histogram figure which shows the similarity/difference
between randomly generated sentences and sentences from our corpus.
"""
import os
import sys
import argparse
import numpy as np
from typing import *
import evaluation.embeddings.util as eeu


def main(args):
	# Generate random sentences.
	random_sentences: List[str] = eeu.random_sentences(args.n, use_corpus=False)

	# Select N sentences from our GT corpus.
	corpus_sentences: List[str] = eeu.random_sentences(args.n, use_corpus=True)

	# Compute embeddings.

	# Compute distances.

	# Plot result.
	pass



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('output_dir', type='str', help='Where to save the output figure.')
	parser.add_argument('--embedding_name', default='word2vec', type=str, choices=['bert', 'word2vec', 'use'])
	parser.add_argument('--distance', default='cosine', type=str, choices=['cosine', 'wasserstein', 'euclidean'])
	parser.add_argument('--n', type=int, help='Number of sentences to use for pairwise comparison.')
	main(parser.parse_args())
