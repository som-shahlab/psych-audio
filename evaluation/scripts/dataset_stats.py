"""
Computes dataset statistics for the abstract.
"""
import os
import sys
import math
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
import soundfile as sf   
import evaluation.util
import evaluation.config as config


def main():
	"""
	Uncomment each of the functions to run them. By default, they are
	"""
	# Load the metadata file.

	# Compute audio average length.
	# audio_lengths()

	# Compute age.
	average_age()

	# Compute number of words.
	# Load the paired file. We want ALL the data.
	paired = evaluation.util.load_paired_json(skip_empty=True)

	pass


def average_age():
	"""
	Computes the average age and stats.
	"""
	df = pd.read_csv(config.META_FQN, sep='\t')
	ages = []
	for _, row in df.iterrows():
		if row['asr_test']:
			age = row['Age_ses1']
			if not math.isnan(age):
				ages.append(age)
	
	print_stats(ages)


def audio_lengths():
	"""
	Computes the audio length, in terms of minutes.
	"""
	FLAC_DIR = '/vol0/psych_audio/jasa_format/v6/flac'
	df = pd.read_csv(config.META_FQN, sep='\t')
	hashes = []
	for _, row in df.iterrows():
		hash_ = row['hash']
		if row['asr_test']:
			hashes.append(hash_)
	
	# Load each audio file and compute length.
	lens = []
	for hash_ in tqdm(hashes, desc='Audio Length'):
		fqn = os.path.join(FLAC_DIR, f'{hash_}.flac')
		data, sr = sf.read(fqn)
		seconds = len(data) / sr
		minutes = seconds / 60
		lens.append(minutes)
	
	lens = np.asarray(lens)
	print('Duration')
	print_stats(lens)


def print_stats(arr: Union[np.ndarray, List]):
	"""
	Prints the mean, standard deviation, median, and range.
	
	Args:
		arr (np.ndarray): Array of values to compute stats over.
	"""
	if len(arr) == 0:
		print('Error: `arr` has length 0.')
		sys.exit(0)
	if not isinstance(arr, np.ndarray):
		arr = np.asarray(arr)

	mean = arr.mean()
	std = arr.std()
	median = np.median(arr)
	low = arr.min()
	high = arr.max()
	
	print(f'Mean:\t{mean}')
	print(f'Std:\t{std}')
	print(f'Median:\t{median}')
	print(f'Min:\t{low}')
	print(f'Max:\t{high}')
	print(f'N:\t{len(arr)}')


if __name__ == '__main__':
	main()
