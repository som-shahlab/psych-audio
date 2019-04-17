"""
This file generates the Table 2 which contains performance metrics broken down by various attributes
such as gender, speaker, etc.
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
SEGMENT_METRICS = '../results/segment.csv'
SESSION_METRICS = '../results/session.csv'


def main(args):
	"""
	Main function which handles program flow, etc.
	:param args: Argparse namespace.
	"""
	if not os.path.exists(META_FQN):
		print(f'File does not exist: {META_FQN}')
		sys.exit(1)

	# Create mapping from hash -> dimensions.
	# Load the metadata file.
	hash2dims, unique_dim_vals = load_dimensions()

	# Load the segment level results.
	segment_results = pd.read_csv(SEGMENT_METRICS, sep=',')
	session_results = pd.read_csv(SESSION_METRICS, sep=',')

	# For each dimension, compute the table rows
	table = []
	for speaker in ['P', 'T']:
		for dim in dimensions:
			# For each unique value, compute the metrics.
			for unique_val in unique_dim_vals[dim]:
				# Grab all segment-level results for this dimension and unique value.
				wers, bleus, gleus = [], [], []
				for _, row in segment_results.iterrows():
					hash_ = row['hash']
					# Error handling. We should always have `hash_` in the meta file.
					if hash_ not in hash2dims:
						print(f'Could not find {hash_} in meta file.')
						continue
					# We may not have dimension information for this hash.
					if dim not in hash2dims[hash_]:
						continue
					if hash2dims[hash_][dim] == unique_val and row['tag'] == speaker:
						wers.append(row['wer'])
						bleus.append(row['bleu'])
						gleus.append(row['gleu'])
				if len(wers) == 0:
					continue
				wers = np.asarray(wers)
				bleus = np.asarray(bleus)
				gleus = np.asarray(gleus)

				table_row = [
					'phrase', speaker, dim, unique_val, len(wers),
					wers.mean(), bleus.mean(), gleus.mean(),
					wers.std(), bleus.std(), gleus.std(),
				]
				table.append(table_row)
				print(table_row)

				# Compute session level results.
				wers, bleus, gleus = [], [], []
				for _, row in session_results.iterrows():
					hash_ = row['hash']
					# Error handling. We should always have `hash_` in the meta file.
					if hash_ not in hash2dims:
						print(f'Could not find {hash_} in meta file.')
						continue
					# We may not have dimension information for this hash.
					if dim not in hash2dims[hash_]:
						continue
					if hash2dims[hash_][dim] == unique_val and row['tag'] == speaker:
						wers.append(row['wer'])
						bleus.append(row['bleu'])
						gleus.append(row['gleu'])
				if len(wers) == 0:
					continue
				wers = np.asarray(wers)
				bleus = np.asarray(bleus)
				gleus = np.asarray(gleus)

				table_row = [
					'session', speaker, dim, unique_val, len(wers),
					wers.mean(), bleus.mean(), gleus.mean(),
					wers.std(), bleus.std(), gleus.std(),
				]
				table.append(table_row)
				print(table_row)

	with open('table2.csv', 'w') as f:
		f.write('type,speaker,dim,val,n,wer_mean,bleu_mean,gleu_mean,wer_std,bleu_std,gleu_std' + '\n')
		for row in table:
			result = ''
			for item in row:
				result += str(item) + ','
			result = result[:-1]
			f.write(result + '\n')


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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	main(parser.parse_args())
