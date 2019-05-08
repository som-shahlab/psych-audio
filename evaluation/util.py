"""
Utility functions for evaluation.
"""
import json
import unidecode

# Ground truth/predictions text file.
PAIRED_FILE = '/vol0/psych_audio/ahaque/psych-audio/results/paired.json'


def load_paired_json(skip_empty=False):
	"""
	Loads the paired json file. Optionally, remove sentences that have a blank GT or pred.
	:return paired: Dictionary of gt/pred/gid/etc values.
	"""
	with open('results/paired.json') as f:
		data = json.load(f)

	# Remove empty sentences if desired.
	if not skip_empty:
		return data
	else:
		paired = {}
		for gid in data.keys():
			if len(data[gid]['gt']) == 0 or len(data[gid]['pred']) == 0:
				continue
			else:
				paired[gid] = data[gid]
		del data
		return paired


def canonicalize(sentence: str) -> str:
	"""
	Converts a sentence into standard, canonicalized format by removing
	punctuation, unicode accent characters, etc.
	
	Args:
		sentence (str): Input sentence.
	
	Returns:
		str: Output, cleaned sentence.
	"""
	sentence = sentence.lower().strip().replace('\'', '')
	sentence = unidecode.unidecode(sentence)
	return sentence
