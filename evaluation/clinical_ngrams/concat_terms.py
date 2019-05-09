"""
Concats all words for a PHQ bucket. Used for generating Table 3.
"""
import os
import sys
import pandas as pd


def main():
	df = pd.read_csv('word2phq.tsv', sep='\t')
	terms = {x: [] for x in range(1, 10)}
	for i, row in df.iterrows():
		phq = int(row['phq'])
		ngram = row['word']
		terms[phq].append(ngram)
	
	for i in range(1, 10):
		print(i, ', '.join(terms[i]))


if __name__ == '__main__':
	main()
