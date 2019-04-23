"""
This file creates a Table 2 (quant results) given a single CSV file of GID,metric pairs.

Prerequisites:
- Result csv files.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any

# Which dimensions to analyze. That is, each item in the below list
# will be a subsection of the table. i.e., Gender will be broken down into male/female (its unique values).
dimensions = [
	'Gender_ses1',
	'Num_sess',
	'Age_ses1',
	'PHQ9_total_ses',
]

#   Path to the meta FQN.
META_FQN = '/vol0/psych_audio/jasa_format/metadata.tsv'


def main(args):
	"""
	Main function which handles program flow, etc.
	:param args: Argparse namespace.
	"""
	if not os.path.exists(META_FQN):
		print(f'File does not exist: {META_FQN}')
		sys.exit(1)
	if not os.path.exists(args.result_csv):
		print(f'File does not exist: {args.result_csv}')
		sys.exit(1)

	# Load the result csv file.
	result = pd.read_csv(args.result_csv, sep=',')

	# Create mapping from hash -> dimensions.
	# Load the metadata file.
	hash2dims, unique_dim_vals = load_dimensions()

	hash2accumulator = sort_metrics_by_hash(result,
	# For each hash, store the phrase-level metrics.
	# At the very end, we'll compute our metrics of interest, just once.
	hash2accumulator: List[str, List[float]] = {}

	# Loop over each gid once.
	for i, row in df.iterrows():
		gid = row['gid']
		val = row['value']
		# Find the associated hash and add it to the running list.


def load_dimensions() -> (Dict[str, Dict[str, int]], Dict[str, List[int]]):
	"""
	Loads the metadata file and returns various dimensions for each hash file.
	:return hash2dims: Dictionary with key=hash, value=Dict of dimensions.
	:return unique_dim_vals: For each dimension, contains the unique values.
	"""
	hash2dims = {}
	unique_dim_vals = {x: set() for x in dimensions}
	df = pd.read_csv(META_FQN, sep='\t')
	for _, row in df.iterrows():
		hash_ = row['hash']
		# Skip hashes already seen.
		if hash_ in hash2dims:
			print(f'Duplicate: {hash_}')
			continue
		# Populate the dimension's values.
		values = {}
		for key in dimensions:
			v = row[key]
			if str(v) == 'nan':
				continue
			else:
				v = int(v)
				values[key] = v
				unique_dim_vals[key].add(v)
		hash2dims[hash_] = values

	return hash2dims, unique_dim_vals


def autodetect_names(data_dir: str) -> List[str]:
	"""
	Automatically detects the embedding names. e.g. bert, word2vec
	:param data_dir: Path to the npz files.
	:return names: List of embedding names.
	"""
	filenames: List[str] = os.listdir(data_dir)
	names: Set[str] = set()
	for filename in filenames:
		key1 = '_gt.npz'
		key2 = '_pred.npz'
		if key1 in filename:
			idx = filename.index(key1)
		else:
			idx = filename.index(key2)
		name = filename[:idx]
		names.add(name)
	names: List[str] = list(names)
	return names


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('result_csv', type=str, help='Location of the result CSV file.')
	main(parser.parse_args())
