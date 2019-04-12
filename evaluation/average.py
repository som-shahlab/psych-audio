import pandas as pd
import matplotlib.pyplot as plt


def main():
	# Compute session-level stats.
	data = {
		'session': pd.read_csv('results/session.csv', sep=','),
		'segment': pd.read_csv('results/segment.csv', sep=','),
	}
	print('Level\tTag\tBLEU\tGLEU\tWER')
	for level in ['session', 'segment']:
		for speaker in ['T', 'P']:
			bleu, gleu, wer = compute_stats(data[level], speaker)
			print(f'{level}\t{speaker}\t{bleu:.2f}\t{gleu:.2f}\t{wer:.2f}')


def compute_stats(df: pd.DataFrame, speaker: str):
	"""
	Computes stats for a particular speaker.

	:param df: DataFrame of segment or session level results.
	:param speaker: 'T' or 'P'
	:return: BLEU, GLEU, WER, etc.
	"""
	assert(speaker in ['T', 'P'])
	idx = (df['tag'] == speaker)
	subset = df[idx]
	bleu = subset['bleu'].mean() * 100
	gleu = subset['gleu'].mean() * 100
	wer = subset['wer'].mean() * 100
	return bleu, gleu, wer


if __name__ == '__main__':
	main()
