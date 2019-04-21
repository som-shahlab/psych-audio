"""
Utility functions for evaluation.
"""
import pandas as pd
from typing import List, Dict

# Path to the CSV file containing phrase-level results. This is used to create the GID/speaker mapping.
PHRASE_RESULTS = '/vol0/psych_audio/ahaque/psych-audio/results/phrase.csv'

# Ground truth/predictions text file.
TEXT_FILE = '/vol0/psych_audio/ahaque/psych-audio/results/text.txt'


def load_gid2speaker():
	# For each GID, find whether it was speaker or therapist.
	df = pd.read_csv(PHRASE_RESULTS, sep=',')
	gid2speaker = []
	for _, row in df.iterrows():
		gid2speaker.append(row['tag'])
	return gid2speaker


def load_gt_pred_text_file(skip_empty=False) -> (Dict[str, List[str]], Dict[str, List[str]]):
	"""
	Loads the text file containing the paired ground truth and predicted transcriptions.
	:param skip_empty: If True, skips any sentence if the pred OR ground truth is empty.
	:return sentences: Dictionary with keys 'gt', 'pred' and value=list of sentences.
	:return gids: Dictionary with keys 'gt', 'pred' and value=List of global IDs.
	"""
	# Load the GT and prediction file.
	sentences: Dict[str, List[str]] = {'gt': [], 'pred': []}
	gids: Dict[str, List[str]] = {'gt': [], 'pred': []}
	with open(TEXT_FILE, 'r') as f:
		lines = f.readlines()

	skip_gids = set()  # skip any pairs where GT or pred is empty.
	for i in range(len(lines)):
		gid, sentence = tuple(lines[i].strip().split(','))
		gid = int(gid)

		if skip_empty:
			if len(sentence) == 0:
				skip_gids.add(gid)
				continue
			if gid in skip_gids:
				continue

		key = 'pred' if i % 2 == 0 else 'gt'
		sentences[key].append(sentence)
		gids[key].append(gid)
	return sentences, gids
