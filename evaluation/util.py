"""
Utility functions for evaluation.
"""
import pandas as pd

# Path to the CSV file containing phrase-level results. This is used to create the GID/speaker mapping.
PHRASE_RESULTS = '/vol0/psych_audio/ahaque/psych-audio/results/phrase.csv'


def load_gid2speaker():
	# For each GID, find whether it was speaker or therapist.
	df = pd.read_csv(PHRASE_RESULTS, sep=',')
	gid2speaker = []
	for _, row in df.iterrows():
		gid2speaker.append(row['tag'])
	return gid2speaker