import pandas as pd
import matplotlib.pyplot as plt


def main():
	df = pd.read_csv('metrics.csv', sep=',')

	avg_bleu = df['bleu'].mean()
	avg_gleu = df['gleu'].mean()
	avg_wer = df['wer'].mean()
	print(f'BLEU: {avg_bleu}')
	print(f'GLEU: {avg_gleu}')
	print(f'WER: {avg_wer}')

	tidx = df['seg_tag'] == 'T'
	therapist_wer = df[tidx]['wer'].mean()
	therapist_bleu = df[tidx]['bleu'].mean()

	patient_wer = df[~tidx]['wer'].mean()
	patient_bleu = df[~tidx]['bleu'].mean()

	print(f'Patient: BLEU: {patient_bleu} WER: {patient_wer}')
	print(f'Therapist: BLEU: {therapist_bleu} WER: {therapist_wer}')


if __name__ == '__main__':
	main()
