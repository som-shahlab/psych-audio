"""
This script is used to transcribe the FULL dataset (aka v5).

Google has a daily quota on how much we can transcribe.
This script checks which files have been completed in the past,
and selects the next set of files to upload and transcribe.
"""
import os
import sys
import shutil
import argparse
from tqdm import tqdm

jasa_dir = '/vol0/psych_audio/jasa_format'

def main(args):
	# Get list of all flac files.
	all_hashes = get_hashes_from_dir(os.path.join(jasa_dir, 'flac'))

	# Check which flac files have been transcribed already.
	completed_hashes = get_hashes_from_dir(os.path.join(jasa_dir, 'v5', 'machine-video'))

	# Select the next N files.
	untranscribed_hashes = list(set(all_hashes).difference(set(completed_hashes)))
	candidate_hashes = untranscribed_hashes[:args.n]

	# Create symlinks to the flac files in the appropriate dir.
	for hash_ in candidate_hashes:
		source_fqn = os.path.join(jasa_dir, 'flac', f'{hash_}.flac')
		dest_fqn = os.path.join(jasa_dir, 'v5', 'flac', f'{hash_}.flac')
		os.symlink(source_fqn, dest_fqn)


def get_hashes_from_dir(path: str):
	"""
	Returns a list of hashes (whether json or flac) from a directory.
	
	Args:
		path (str): Path to a folder containing json or flac files.
	"""
	filenames = sorted(os.listdir(path))
	hashes = [x.replace('.flac', '') for x in filenames]
	hashes = [x.replace('.json', '') for x in hashes]
	return hashes


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('n', type=int, help='Number of files to queue for upload.')
	args = parser.parse_args()
	main(args)
