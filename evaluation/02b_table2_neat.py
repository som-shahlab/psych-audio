"""
Performs significance testing for patient/therapist and male/female
for both WER and EMD.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import *
import evaluation.config


def main():
    # df = pd.read_csv(evaluation.config.TABLE2_FQN, sep="\t")
    df = pd.read_csv("table2.tsv", sep="\t")

    # Compute therapist vs patient stats.
    analyze_speakers(df)

    # Compute male vs female stats.
    analyze_genders(df)


def analyze_speakers(df: pd.DataFrame):
    """
    Gets metrics for patients and therapists, then computes stats.

    Args:
        df (pd.DataFrame): Raw table 2 data.
    """
    for metric in ["WER", "EMD"]:
        patient = []
        therapist = []
        for _, row in df.iterrows():
            speaker = row["speaker"]
            if speaker == "P":
                patient.append(row[metric])
            elif speaker == "T":
                therapist.append(row[metric])


def analyze_genders(df: pd.DataFrame):
    """
    Gets metrics for male and female patients, then computes stats.

    Args:
        df (pd.DataFrame): Raw table 2 data.
    """
    raise NotImplementedError


def difference_test(arr1: np.ndarray, arr2: np.ndarray) -> Dict:
    """
    Given two arrays, each array containing numbers from two different samples,
    compute the two-tailed independent t-test.
    
    Args:
        arr1 (np.ndarray): Samples from group 1.
        arr2 (np.ndarray): Samples from group 2.
    
    Returns:
        Dict: Struct containing confidence intervals, p-values, etc.
    """


if __name__ == "__main__":
    main()
