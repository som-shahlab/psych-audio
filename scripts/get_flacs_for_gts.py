"""
This script looks at a folder of ground truth transcriptions,
finds the associated flac files, then copies these flac files into a new directory.
"""
import os
import sys
import shutil
import argparse


def main(args):
    # Get list of json filenames.
    ls = os.listdir(args.gt_dir)
    hashes = [x.replace(".json", "") for x in ls]

    for hash in hashes:
        source_fqn = os.path.join(args.source_flac_dir, f"{hash}.flac")
        dest_fqn = os.path.join(args.dest_flac_dir, f"{hash}.flac")
        shutil.copy(source_fqn, dest_fqn)
        print(dest_fqn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gt_dir",
        type=str,
        help="Folder containing ground truth json files with hashed filenames.",
    )
    parser.add_argument(
        "source_flac_dir", type=str, help="Location of the source flac files."
    )
    parser.add_argument(
        "dest_flac_dir", type=str, help="Location to place the new flac files."
    )
    args = parser.parse_args()
    main(args)
