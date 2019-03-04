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


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hi')
    args = parser.parse_args()
    main(args)
