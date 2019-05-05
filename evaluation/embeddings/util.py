"""
Contains various util functions for loading and computing embeddings.
"""
import gensim
import numpy as np
from evaluation.embeddings import config
from typing import *


def load_embedding_model(embedding_name: str) -> (Dict, Set[str]):
	"""
	Loads an embedding model.
	
	Args:
		embedding_name (str): 'glove' or 'word2vec'
	Returns:
		model: Dictionray of key->np.ndarray embeddings
		keys: Set of strings, corresponding to all words in the vocab.
	"""
	if embedding_name == 'word2vec':
		model = gensim.models.KeyedVectors.load_word2vec_format(config.WORD2VEC_MODEL_FQN, binary=True)
		keys = model.vocab
	elif embedding_name == 'glove':
		model = load_glove(config.GLOVE_MODEL_FQN)
		keys = set(model.keys())
	return model, keys


def encode_from_dict(embedding_name: str, model: Dict[str, np.ndarray], keys: Set[str], sentence: str) -> Optional[np.ndarray]:
	"""
	Encodes a sentence using a dictionary-based embedding. That is, either word2vec or Glove.
	A dictionary-based embedding has words as the keys and a numpy array as the value.

	:param embedding_name: Embedding name.
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
	embeddings = np.zeros((count, config.F[embedding_name]), np.float32)
	idx = 0
	for word in words:
		if word in keys:
			embeddings[idx] = model[word]
			idx += 1

	# Mean pooling.
	embedding = embeddings.mean(0)
	return embedding


def load_glove(path: str) -> Dict[str, np.ndarray]:
	"""
	Loads the GloVe model.
	
	Args:
		path (str): Full path to the glove model (text file).
	
	Returns:
		Dict[str, np.ndarray]: Glove model with key=word and value=embedding vector.
	"""
	model: Dict[str, np.ndarray] = {}
	with open(path, 'r') as f:
		for line in f:
			tokens = line.strip().split(' ')
			word = tokens[0]
			vec = np.array([float(x) for x in tokens[1:]])
			model[word] = vec
	return model
