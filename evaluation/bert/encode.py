"""
This file computes the BERT embeddings for GT and predictions.

Prerequisites:
- Transcriptions text file (text.txt) resulting from `evaluation/phrase_level.py`.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import evaluation.util
from bert_serving.client import BertClient


max_words = 50


def main(args):
	if not os.path.exists(args.text_file):
		print(f'File does not exist: {args.text_file}')
		sys.exit(1)

	print('Creating BERT client...')
	bc = BertClient(check_length=False)

	sentences, gids = evaluation.util.load_gt_pred_text_file(skip_empty=True)
	pred_sentences = sentences['pred']
	pred_gids = gids['pred']
	gt_sentences = sentences['gt']
	gt_gids = gids['gt']

	if args.n_lines > 0:
		pred_sentences = pred_sentences[:args.n_lines]
		pred_gids = np.asarray(pred_gids[:args.n_lines])
		gt_sentences = gt_sentences[:args.n_lines]
		gt_gids = np.asarray(gt_gids[:args.n_lines])

	# Get embeddings.
	print(f'Encoding {len(gt_gids)} sentences...')
	pred_embeddings = bc.encode(pred_sentences)
	gt_embeddings = bc.encode(gt_sentences)

	print('Converting text to numpy...')
	pred_sentences = np.asarray(pred_sentences)
	gt_sentences = np.asarray(gt_sentences)

	# Write embeddings to file.
	print('Saving BERT preds...')
	np.savez('bert_pred.npz', embeddings=pred_embeddings, gids=pred_gids, text=pred_sentences)
	print('Saving BERT ground truths...')
	np.savez('bert_gt.npz', embeddings=gt_embeddings, gids=gt_gids, text=gt_sentences)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_lines', type=int, default=-1, help='Number of lines to process. Use -1 to process all.')
	main(parser.parse_args())
