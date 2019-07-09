"""Computes various statistics."""
import scipy.stats
import numpy as np
from typing import *


def difference_test(labels: List, arr0: np.ndarray, arr1: np.ndarray) -> Dict:
    """
    Given two arrays, each array containing numbers from two different samples,
    compute the two-tailed independent t-test.
    
    Args:
        arr0 (np.ndarray): Samples from group 0.
        arr1 (np.ndarray): Samples from group 1.
    
    Returns:
        Dict: Struct containing confidence intervals, p-values, etc.
    """
    # Test whether arr0 and arr1 are normally distributed.
    # Compute t-test/p-values.
    statistic0, pvalue0 = scipy.stats.shapiro(arr0)
    statistic1, pvalue1 = scipy.stats.shapiro(arr1)

    print("------------ NORMALITY TESTS ------------")
    print("Shapiro-Wilk")
    print(f"\t{labels[0]} w: {statistic0:.4f}\tP: {pvalue0:.3e}")
    print(f"\t{labels[1]} w: {statistic1:.4f}\tP: {pvalue1:.3e}")

    print("------------ VARIANCE TESTS ------------")
    print("Bartlett (Normally Distributed)")
    stat, pval = scipy.stats.bartlett(arr0, arr1)
    print(f"\tT: {stat:.4f}\tP: {pval:.3e}")

    print("Levene (Not Normally Distributed)")
    stat, pval = scipy.stats.levene(arr0, arr1)
    print(f"\tw: {stat:.4f}\tP: {pval:.3e}")

    print("------------ DIFFERENCE TESTS ------------")
    print("Student t-test (Equal Var; Normally Distributed)")
    stat, pval = scipy.stats.ttest_ind(arr0, arr1, equal_var=True)
    print(f"\tt: {stat:.4f}\tP: {pval:.3e}")

    print("Welch's t-test (Unequal Var; Normally Distributed)")
    stat, pval = scipy.stats.ttest_ind(arr0, arr1, equal_var=False)
    print(f"\tt: {stat:.4f}\tP: {pval:.3e}")

    print("Mann-Whitney U-test (Not Normally Distributed)")
    stat, pval = scipy.stats.mannwhitneyu(arr0, arr1)
    print(f"\tu: {stat:.4f}\tP: {pval:.3e}")
