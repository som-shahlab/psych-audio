"""
Loads the cropped sentences (tsv files) and computes WER.
"""
import os
import sys
import nltk
import argparse
import numpy as np
import pandas as pd
import evaluation.util
from evaluation.self_harm import config

def main():
	# Load the cropped files.
	filenames = os.listdir(config.OUT_DIR)

	metrics = {
		'BLEU': [],
		'WER': [],
	}
	for filename in filenames:
		fqn = os.path.join(config.OUT_DIR, filename)
		df = pd.read_csv(fqn, sep='\t')
		
		# Compose the GT and pred sentences.
		gt = []
		pred = []
		conf = []
		for i, row in df.iterrows():
			gt_word = row['gt']
			pred_word = row['pred']

			# Check if nan.
			if isinstance(gt_word, float):
				gt_word = ''
			if isinstance(pred_word, float):
				pred_word = ''

			gt.append(gt_word)
			pred.append(pred_word)
			conf.append(row['conf'])

		gt = ' '.join(gt)
		pred = ' '.join(pred)
		bleu = nltk.translate.bleu_score.sentence_bleu([gt], pred)
		wer = evaluation.util.word_error_rate(pred, gt)

		print('-' * 80)
		print(f'GT:\t{gt}')
		print(f'Pred:\t{pred}')
		conf_str = ''
		for x in conf:
			conf_str += f' {x:.2f}'
		print(conf_str.strip())
		print(f'WER: {wer:.4f}')
		print(f'BLEU: {bleu:.4f}')

		metrics['WER'].append(wer)
		metrics['BLEU'].append(bleu)

	print('------ FINAL METRICS ------')
	bleus = np.asarray(metrics['BLEU'])
	wers = np.asarray(metrics['WER'])
	print(f'n: {len(bleus)}')
	print(f'WER:\t{wers.mean():.4f} +/- {wers.std():.4f}')
	print(f'BLEU:\t{bleus.mean():.4f} +/- {bleus.std():.4f}')


if __name__ == '__main__':
	main()
