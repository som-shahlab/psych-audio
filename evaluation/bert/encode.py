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
from bert_serving.client import BertClient


max_words = 50


def main(args):
	if not os.path.exists(args.text_file):
		print(f'File does not exist: {args.text_file}')
		sys.exit(1)
	if args.n_lines <= 0:
		print('Invalid n_lines')
		sys.exit(1)

	print('Creating BERT client...')
	bc = BertClient(check_length=False)

	# Load the GT and prediction file.
	print('Loading file...')
	pred_sentences, pred_gids = [], []
	gt_sentences, gt_gids = [], []
	with open(args.text_file, 'r') as f:
		lines = f.readlines()

	print('Creating arrays...')
	# BERT does not allow empty strings. Therefore we need to keep track of which idx each gt/pred belongs to.
	for i in range(len(lines)):
		is_pred = False
		if i % 2 == 0:
			is_pred = True

		gid, sentence = tuple(lines[i].strip().split(','))
		if len(sentence) == 0:
			continue

		if is_pred:
			pred_sentences.append(sentence)
			pred_gids.append(gid)
		else:
			gt_sentences.append(sentence)
			gt_gids.append(gid)

	pred_sentences = pred_sentences[:args.n_lines]
	pred_gids = np.asarray(pred_gids[:args.n_lines])
	gt_sentences = gt_sentences[:args.n_lines]
	gt_gids = np.asarray(gt_gids[:args.n_lines])

	# Get embeddings.
	print('Encoding preds...')
	pred_embeddings = bc.encode(pred_sentences)
	gt_embeddings = bc.encode(gt_sentences)

	print('Converting text to numpy...')
	pred_sentences = np.asarray(pred_sentences)
	gt_sentences = np.asarray(gt_sentences)

	# Write embeddings to file.
	print('Saving BERT preds...')
	np.savez('bert_preds.npz', bert=pred_embeddings, gids=pred_gids, text=pred_sentences)
	print('Saving BERT ground truths...')
	np.savez('bert_gts.npz', bert=gt_embeddings, gids=gt_gids, text=gt_sentences)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('text_file', type=str, help='Location of the combined GT/pred text file.')
	parser.add_argument('n_lines', type=int, help='Number of lines to process.')
	main(parser.parse_args())
