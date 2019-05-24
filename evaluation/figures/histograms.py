"""
Creates the histogram figures and computes

Each plot contains one histogram for each dimension value (e.g., male vs female).
It also shows the perfect and random performance.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import *
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

METRICS = ['WER', 'BLEU', 'COSINE', 'EMD']
X_AXIS_LABEL = {
	'WER': 'Word Error Rate',
	'BLEU': 'BLEU Score',
	'COSINE': 'Cosine Distance',
	'EMD': 'Earth Mover Distance',
}

DARKBLUE = (0.184, 0.341, 0.388)
LIGHTBLUE = (0.741, 0.820, 0.855)
BLACK = (0, 0, 0)

def main():
	df = pd.read_csv('results/table2.tsv', sep='\t')
	
	# Compute gender plot.
	for metric in METRICS:
		perfect_perf = 1 if metric == 'BLEU' else 0
		male = df[df['gender'] == 'Male'][metric].values
		female = df[df['gender'] == 'Female'][metric].values
		rand_perf = df[f'RAND_{metric}'].values.mean()
		out_fqn = os.path.join('results', f'hist_gender_{metric}.png')
		create_dual_hist(metric, out_fqn, male, female, rand_perf, perfect_perf, labels=['Male', 'Female'])
		break
		# therapist = df[df['speaker'] == 'T'][metric].values
		# patient = df[df['speaker'] == 'P'][metric].values
		# out_fqn = os.path.join('results', f'hist_speaker_{metric}.png')
		# create_dual_hist(metric, out_fqn, therapist, patient, rand_perf, perfect_perf, labels=['Therapist', 'Patient'])



def create_dual_hist(metric, out_fqn: str, arr1: np.ndarray, arr2: np.ndarray, rand_perf, perfect_perf, labels: List[str]):
	"""
	Creates a single plot with two histograms.
	
	Args:
		out_fqn: Location to save the figure.
		arr1 (np.ndarray): Dataset 1 of values.
		arr2 (np.ndarray): Dataset 2 of values.
		labels (List[str]): List of labels to use for the legend.
	"""
	n_bins = 25
	line_thickness = 3

	fig, axes = plt.subplots(
		# figsize=(width, height)
		nrows=3, ncols=1, figsize=(10, 8), sharex='col',
		gridspec_kw={'height_ratios': [1, 1, 1]}
	)

	# Need to scale the histogram and PDF to match the count.
	max_val = max(arr1.max(), arr2.max(), 1.0, rand_perf)
	x1, y1 = fit_normal_line(arr1, n_bins, max_val)
	x2, y2 = fit_normal_line(arr2, n_bins, max_val)
	margin = 0.07  # As percent of the horizontal area.
	min_val = 0 - max_val * margin
	max_val = max_val * (1 + margin)  # Add margin on the right side.

	axes[0].hist(arr1, n_bins, fc=DARKBLUE, label=labels[0])
	axes[0].set_xlim([min_val, max_val])
	axes[0].set(ylabel='# Sessions')
	axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	axes[0].legend()

	axes[1].hist(arr2, n_bins, fc=LIGHTBLUE, label=labels[1])
	axes[1].set(ylabel='# Sessions')
	axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	axes[1].legend()

	axes[2].plot(x1, y1, '-', c=DARKBLUE, linewidth=line_thickness)
	axes[2].plot(x2, y2, '-', c=LIGHTBLUE, linewidth=line_thickness)
	axes[2].set(xlabel=X_AXIS_LABEL[metric], ylabel='Frequency')
	axes[2].set_yticklabels(['{:,.0%}'.format(x) for x in axes[2].get_yticks()])
	if metric in ['WER', 'BLEU']:
		axes[2].set_xticklabels(['{:,.0%}'.format(x) for x in axes[2].get_xticks()])
	else:
		axes[2].set_xticklabels(['{:,.1f}'.format(x) for x in axes[2].get_xticks()])

	# Add vertial lines/bounds.
	axes[2].vlines(x=perfect_perf, ymin=0, ymax=max(y1.max(), y2.max()) * .8, linestyles='dashed', color='k')
	axes[2].vlines(x=rand_perf, ymin=0, ymax=max(y1.max(), y2.max()) * .8, linestyles='dashed', color='k')

	plt.savefig(out_fqn, bbox_inches='tight')
	print(out_fqn)
	statistical_tests(arr1, arr2)


def statistical_tests(arr1: np.ndarray, arr2: np.ndarray):
	"""Runs our suite of statistial test."""
	out_fqn = 'qq.png'
	fig, axes = plt.subplots(
		# figsize=(width, height)
		nrows=1, ncols=2, figsize=(16, 7), sharex='col',
		gridspec_kw={'width_ratios': [1, 1]}
	)

	# Compute t-test/p-values.
	scipy.stats.probplot(arr1, plot=axes[0])
	scipy.stats.probplot(arr2, plot=axes[1])
	statistic1, pvalue1 = scipy.stats.shapiro(arr1)
	statistic2, pvalue2 = scipy.stats.shapiro(arr2)
	print(f'Arr1: Statistic: {statistic1:.4f}\tP-Value: {pvalue1:.4f}')
	print(f'Arr2: Statistic: {statistic2:.4f}\tP-Value: {pvalue2:.4f}')
	plt.savefig(out_fqn)
	print(out_fqn)


def fit_normal_line(arr: np.ndarray, n_bins: int, max_val) -> (np.ndarray, np.ndarray):
	"""
	Fits a normal distribution to the histogram.
	
	Args:
		arr (np.ndarray): Array of original values. This will be fed into the plt.hist/np.histogram function.
		n_bins (int): Number of bins to use.
	
	Returns:
		x: X values of the fitted line.
		y: Y values of the fitted line.
	"""
	mu, sigma = arr.mean(), arr.std()
	x = np.arange(0, max_val, 0.001)
	y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))

	# Need to convert from normal distribution (PDF) into the histogram count version.
	# To do so, we need to compute a scaling factor. We can compute the scaling factor
	# by looking at np.histogram with and without the density=True arg.
	vals_original, _ = np.histogram(arr, n_bins)
	vals_density, _ = np.histogram(arr, n_bins, density=True)
	scaling = vals_original[0] / vals_density[0]  # Un-normalized density -> counts
	scaling = scaling / vals_original.sum()  # counts -> PDF.
	y = y * scaling
	return x, y


if __name__ == '__main__':
	# plt.style.use('ggplot')
	plt.rcParams.update({'font.size': 20})
	main()
