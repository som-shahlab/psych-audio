"""
This file computes the BERT embeddings for GT and predictions.

Prerequisites:
- Transcriptions text file (text.txt) resulting from `evaluation/01_extract_phrases.py`.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import evaluation.util
from bert_serving.client import BertClient

# Max sequence length (words), as specified in `server/start.sh`.
SEQ_LEN = 100


def main(args):
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

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
	# Size: (N, SEQ_LEN, F)
	# N: Number of sentences
	# SEQ_LEN: Max sentence length.
	# F: BERT feature size.
	pred_embeddings = bc.encode(pred_sentences)
	gt_embeddings = bc.encode(gt_sentences)

	print('Converting text to numpy...')
	pred_sentences = np.asarray(pred_sentences)
	gt_sentences = np.asarray(gt_sentences)

	print(pred_embeddings.shape)

	# Write embeddings to file.
	print('Saving embeddings...')
	pred_fqn = os.path.join(args.output_dir, 'bert_pred.npz')
	gt_fqn = os.path.join(args.output_dir, 'bert_gt.npz')
	np.savez_compressed(pred_fqn, embeddings=pred_embeddings, gids=pred_gids, text=pred_sentences)
	np.savez_compressed(gt_fqn, embeddings=gt_embeddings, gids=gt_gids, text=gt_sentences)
	print(pred_fqn)
	print(gt_fqn)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('output_dir', type=str, help='Location to save the output npz files.')
	parser.add_argument('--n_lines', type=int, default=-1, help='Number of lines to process. Use -1 to process all.')
	main(parser.parse_args())
