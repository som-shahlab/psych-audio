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

	# Number of words and talking time.
	num_words()
	

def num_words():
	"""
	Computes the number of words spoken by the therapist vs the patient and
	also computes the talking time (in minutes).
	"""
	# Load the GT.
	df = pd.read_csv(config.META_FQN, sep='\t')
	stats = {
		'T': {'words': [], 'duration': []},
		'P': {'words': [], 'duration': []},
		'sess': {'words': [], 'duration': []}
	}

	for _, row in df.iterrows():
		if row['asr_test']:
			stats['P']['words'].append(float(row['gt_patient_num_words']))
			stats['T']['words'].append(float(row['gt_therapist_num_words']))
			stats['P']['duration'].append(float(row['gt_patient_time_spoken']))
			stats['T']['duration'].append(float(row['gt_therapist_time_spoken']))
			stats['sess']['duration'].append(float(row['sess_dur']))
			n_words = row['gt_therapist_num_words'] + row['gt_patient_num_words']
			stats['sess']['words'].append(n_words)

	for speaker in stats:
		for metric in stats[speaker]:
			print(f'------ {speaker} | {metric} ------')
			print_stats(stats[speaker][metric])


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
	
	print('------ Age ------')
	print_stats(ages)


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
