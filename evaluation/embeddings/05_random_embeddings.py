"""
This script generates random sentences and analyzes the embedding distances.

This script must be run on a linux system.
"""
import os
import sys
import argparse
import numpy as np
from typing import *


word_file = '/usr/share/dict/words'


def main(args):
	# Load the vocabulary.
	with open(word_file, 'r') as f:
		vocab = [x.strip().lower() for x in f.readlines()]

	# Generate N sentences.
	sentences = []
	for _ in range(args.n):
		s = generate_sentence(vocab)
		sentences.append(s)

	print(sentences)


def generate_sentence(vocab: List[str]) -> str:
	"""
	Generates a random English sentence.
	:return sentence: Generated sentence.
	"""
	sentence = ''
	return sentence


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('n', type=int, default=100, help='Number of sentences to generate.')
	main(parser.parse_args())
