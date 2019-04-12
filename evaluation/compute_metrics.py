import pandas as pd
import matplotlib.pyplot as plt


def main():
	# Compute session-level stats.
	data = {
		'session': pd.read_csv('results/session.csv', sep=','),
		'segment': pd.read_csv('results/segment.csv', sep=','),
	}

	f, axarr = plt.subplots(4, 3)
	# Hide x labels and tick labels for top plots and y ticks for right plots.
	for ax in axarr.flat:
		ax.label_outer()

	metric_names = ['BLEU', 'GLEU', 'WER']
	for k in range(len(metric_names)):
		axarr[0, k].set_title(metric_names[k])

	print('Level\tTag\tBLEU\tGLEU\tWER')
	for i, level in enumerate(['segment', 'session']):
		for j, speaker in enumerate(['T', 'P']):
			row = i * 2 + j
			result = get_subset(data[level], speaker)
			label = f'{level}-{speaker}'
			n_bins = 20
			axarr[row, 0].set(ylabel=label)
			for k in range(len(metric_names)):
				axarr[row, k].hist(result[metric_names[k]]['values'], bins=n_bins)

			# Result strings.
			result_string = f'{level}\t{speaker}'
			for metric in metric_names:
				mean = result[metric]['mean']
				std = result[metric]['std']
				result_string += f'\t{mean:.2f} ± {std:.2f}'
			print(result_string)
	plt.savefig('results/plots.png')


def get_subset(df: pd.DataFrame, speaker: str):
	"""
	Returns raw values for a particular speaker.

	:param df: DataFrame of segment or session level results.
	:param speaker: 'T' or 'P'
	:return: BLEU, GLEU, WER, etc.
	"""
	assert(speaker in ['T', 'P'])
	idx = (df['tag'] == speaker)
	subset = df[idx]

	result = {
		'BLEU': {
			'values': subset['bleu'] * 100,
			'std': subset['bleu'].std() * 100,
			'mean': subset['bleu'].mean() * 100,
		},
		'GLEU': {
			'values': subset['gleu'] * 100,
			'std': subset['gleu'].std() * 100,
			'mean': subset['gleu'].mean() * 100,
		},
		'WER': {
			'values': subset['wer'] * 100,
			'std': subset['wer'].std() * 100,
			'mean': subset['wer'].mean() * 100,
		}

	}
	return result


if __name__ == '__main__':
	plt.style.use('ggplot')
	main()
