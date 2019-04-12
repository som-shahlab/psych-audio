import pandas as pd
import matplotlib.pyplot as plt


def main():
	df = pd.read_csv('metrics.csv', sep=',')
	print(df.mean())
	tidx = df['seg_tag'] == 'T'
	therapist_wer = df[tidx]['wer'].mean()
	therapist_bleu = df[tidx]['bleu'].mean()

	patient_wer = df[~tidx]['wer'].mean()
	patient_bleu = df[~tidx]['bleu'].mean()

	print(f'Patient: BLEU: {patient_bleu} WER: {patient_wer}')
	print(f'Therapist: BLEU: {therapist_bleu} WER: {therapist_wer}')

	plt.hist(df[tidx]['wer'], 25, facecolor='g', alpha=0.75)
	plt.show()


if __name__ == '__main__':
	main()
