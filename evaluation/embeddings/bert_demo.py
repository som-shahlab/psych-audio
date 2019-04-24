"""Runs a demo showing BERT distance working."""
import numpy as np
import scipy.spatial.distance


def main():
	gt = np.load('bert_gts.npz')
	gt_text = gt['text']
	pred = np.load('bert_preds.npz')
	pred_text = pred['text']
	for i in range(len(gt_text)):
		cosine = scipy.spatial.distance.cosine(gt['bert'][i], pred['bert'][i])
		print('-' * 40, gt['gids'][i])
		print(f'Pred | {pred_text[i]}')
		print(f'GT   | {gt_text[i]}')
		print(f'Dist | {cosine:.4f}')


if __name__ == '__main__':
	main()
