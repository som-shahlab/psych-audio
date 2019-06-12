"""
This script searches the paired.json file for
examples that contain a specific word.
"""
import os
import sys
import evaluation.util

keywords = ['suicide', 'kill', 'death', 'dead', 'die', 'harm',
			'depression', 'died', 'hurt', 'depressed']

def main():
	# Load the data.
	paired = evaluation.util.load_paired_json(skip_empty=False)

	# Find examples where a keyword appears in the GT or Pred, but not both.
	for gid, v in paired.items():
		if v['speaker'] != 'P':
			continue
		for keyword in keywords:
			in_gt, in_pred = False, False
			if keyword in v['gt'].split(' '):
				in_gt = True
			if keyword in v['pred'].split(' '):
				in_pred = True
			
			if in_gt or in_pred:  # XOR.
				print('-' * 20, v['speaker'], gid, '-' * 20)
				gt = v['gt'].replace(keyword, f'**{keyword.upper()}**')
				pred = v['pred'].replace(keyword, f'**{keyword.upper()}**')
				# Create a prefix so we can easily identify which line
				# contains the keyword.
				gt_prefix = '*' if in_gt else ''
				pred_prefix = '*' if in_pred else ''
				print(f'{gt_prefix}GT:\t{gt}')
				print(f'{pred_prefix}Pred:\t{pred}')
			


if __name__ == '__main__':
	main()
