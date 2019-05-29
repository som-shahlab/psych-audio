"""
Loads the cropped sentences (tsv files) and computes WER.
"""
import os
import sys
import nltk
import argparse
import pandas as pd
import evaluation.util
from evaluation.self_harm import config

def main():
	# Load the cropped files.
	filenames = os.listdir(config.OUT_DIR)

	for filename in filenames:
		fqn = os.path.join(config.OUT_DIR, filename)
		df = pd.read_csv(fqn, sep='\t')
		
		# Compose the GT and pred sentences.
		gt = []
		pred = []
		for i, row in df.iterrows():
			gt.append(row['gt'])
			pred.append(row['pred'])

			bleu = nltk.translate.bleu_score.sentence_bleu(
				references=[gt], hypothesis=pred)
			wer = evaluation.util.word_error_rate(pred, gt)
			rand_bleu = nltk.translate.bleu_score.sentence_bleu(
				references=[gt], hypothesis=random_sentence)
			rand_wer = evaluation.util.word_error_rate(random_sentence, gt)

if __name__ == '__main__':
	main()
