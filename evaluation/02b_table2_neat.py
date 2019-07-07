"""
Performs significance testing for patient/therapist and male/female
for both WER and EMD.
"""
import os
import sys
import scipy.stats
import numpy as np
import pandas as pd
from typing import *
import evaluation.config


def main():
    df = pd.read_csv(evaluation.config.TABLE2_FQN, sep="\t")

    # Compute aggregate.
    compute_aggregate(df)

    # Compute therapist vs patient stats.
    analyze_speakers(df)

    # Compute male vs female stats.
    analyze_genders(df)


def compute_aggregate(df: pd.DataFrame):
    """
    Computes aggregate-level metrics.
    
    Args:
        df (pd.DataFrame): Raw table 2 data.
    """
    for metric in ["WER", "RAND_WER", "EMD", "RAND_EMD"]:
        vals = df[metric]
        mean = vals.mean()
        std = vals.std()
        median = np.median(vals)
        min_ = np.min(vals)
        max_ = np.max(vals)
        print(f"----------- {metric} -----------")
        print(
            f"average (SD) of {mean:.2f} ({std:.2f}) units (median"
            + f" [range]; {median:.2f} [{min_:.2f}-{max_:.2f}])"
        )
        stat, pval = scipy.stats.shapiro(vals)
        print(f"Shapiro-Wilk W={stat:.4f}\tP={pval:.3e}")


def analyze_speakers(df: pd.DataFrame):
    """
    Gets metrics for patients and therapists, then computes stats.

    Args:
        df (pd.DataFrame): Raw table 2 data.
    """
    for metric in ["WER", "EMD"]:
        print(f"------------ Speakers: {metric} ------------")
        patient = []
        therapist = []
        random_p = []
        random_t = []
        for _, row in df.iterrows():
            speaker = row["speaker"]
            if speaker == "P":
                patient.append(row[metric])
                random_p.append(row[f"RAND_{metric}"])
            elif speaker == "T":
                therapist.append(row[metric])
                random_t.append(row[f"RAND_{metric}"])

        print(f"ASR {metric}")
        print(f"\tPatient: {mean_std_string(patient)}")
        print(f"\tTherapist: {mean_std_string(therapist)}")

        print(f"Random {metric}")
        print(f"\tPatient: {mean_std_string(random_p)}")
        print(f"\tTherapist: {mean_std_string(random_t)}")

        patient = np.asarray(patient)
        therapist = np.asarray(therapist)

        difference_test(["P", "T"], patient, therapist)


def mean_std_string(arr: Union[List, np.ndarray]) -> str:
    """
    Given an array of numbers, computes the mean and std and returns
    a string in the format mean ± std.
    
    Args:
        arr (Union[List, np.ndarray]): List of numbers.
    Returns:
        result (str): Mean and std in format `mean ± std`.
    """
    arr = np.asarray(arr)
    return f"{arr.mean():.2f} ± {arr.std():.2f}"


def analyze_genders(df: pd.DataFrame):
    """
    Gets metrics for male and female patients, then computes stats.

    Args:
        df (pd.DataFrame): Raw table 2 data.
    """
    for metric in ["WER", "EMD"]:
        print(f"------------ Genders: {metric} ------------")
        male = []
        female = []
        random_male = []
        random_female = []
        for _, row in df.iterrows():
            if row["speaker"] == "P":
                gender = row["gender"]
                if gender == "Male":
                    male.append(row[metric])
                    random_male.append(row[f"RAND_{metric}"])
                elif gender == "Female":
                    female.append(row[metric])
                    random_female.append(row[f"RAND_{metric}"])

        print(f"ASR {metric}")
        print(f"\tMale: {mean_std_string(male)}")
        print(f"\tFeale: {mean_std_string(female)}")

        print(f"Random {metric}")
        print(f"\tMale: {mean_std_string(random_male)}")
        print(f"\tFemale: {mean_std_string(random_female)}")

        male = np.asarray(male)
        female = np.asarray(female)

        difference_test(["M", "F"], male, female)


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

    print("Shapiro-Wilk")
    print(f"\t{labels[0]} w: {statistic0:.4f}\tP: {pvalue0:.3e}")
    print(f"\t{labels[1]} w: {statistic1:.4f}\tP: {pvalue1:.3e}")

    print("Students t-test")
    stat, pval = scipy.stats.ttest_ind(arr0, arr1)
    print(f"\tt: {stat:.4f}\tP: {pval:.3e}")

    print("Mann-Whitney U-test")
    stat, pval = scipy.stats.mannwhitneyu(arr0, arr1)
    print(f"\tu: {stat:.4f}\tP: {pval:.3e}")


if __name__ == "__main__":
    main()
