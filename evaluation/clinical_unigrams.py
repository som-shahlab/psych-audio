"""
This script computes the clinical unigram score between the GT and prediction.
"""
import os
import sys
import argparse
import numpy as np


def main(args):
    # Load the GT and prediction file.
    if not os.path.exists(args.text_file):
        print(f'File does not exist: {args.text_file}')
        sys.exit(1)

    print('Loading file...')
    pred_sentences, pred_gids = [], []
    gt_sentences, gt_gids = [], []
    with open(args.text_file, 'r') as f:
        lines = f.readlines()

    print('Creating arrays...')
    for i in range(len(lines)):
        gid, sentence = tuple(lines[i].strip().split(','))
        if i % 2 == 0:
            pred_sentences.append(sentence)
            pred_gids.append(gid)
        else:
            gt_sentences.append(sentence)
            gt_gids.append(gid)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text_file', type=str, help='Location of the combined GT/pred text file.')
    main(parser.parse_args())
