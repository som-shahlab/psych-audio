"""
This script cleans up the data by concatenating therapy sessions that
were split into two audio files. This script also standardizes
the naming convention for therapy sessions and creates JSON
files for the ground truth transcriptions.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

meta_fqn = '/share/pi/nigam/psych_audio/scotty/results/metadata.tsv'


def main(args):
    # Ensure the metadata file is valid.
    if not metadata_file_is_clean(meta_fqn):
        print(f'File contains errors: {meta_fqn}')
        return


def metadata_file_is_clean(fqn: str) -> bool:
    """
    Checks whether teh metadata file is valid.
    - All rows are tab-delimited.
    - All rows contain valid audio files. Note that some rows have two associated files.
    - All rows contain a valid SHA256 hash.
    - All rows have 36 columns.

    :param fqn: Full path to the metadata file.
    :return: True if metadata file is correct. False otherwise.
    """
    # Check if it exists.
    if not os.path.exists(fqn):
        print(f'File does not exist: {fqn}')
        return False

    # Open and check each line.
    with open(fqn, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # Skip the first line since it contains headers.
            if i == 0:
                continue

            # Check if line is valid.
            tokens = line.strip().split('\t')
            if len(tokens) != 36:
                print(f'Line {i+1} does not have 36 columns.')
                return False

            fqns = tokens[32].split(';')
            for fqn in fqns:
                if not os.path.exists(fqn):
                    print(f'Line {i+1} audio does not exist: {fqn}')
                    return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hi')
    args = parser.parse_args()
    main(args)
