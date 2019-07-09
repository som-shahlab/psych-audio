"""
Checkes whether the flac files for the v4 and v6
dataset are exactly the same (including content).

This script is a one-time use script that is used
to create the v6 dataset.
"""
import os
import sys
import hashlib
import shutil
import argparse
from tqdm import tqdm

jasa_dir = "/vol0/psych_audio/jasa_format"


def main(args):
    v4 = sorted(os.listdir(os.path.join(jasa_dir, "v4", "flac")))
    v6 = sorted(os.listdir(os.path.join(jasa_dir, "v6", "flac")))

    # Check filenames.
    for filename in v4:
        if filename not in v6:
            print(f"In v4 but not v6: {filename}")

    for filename in v6:
        if filename not in v4:
            print(f"In v6 but not v4: {filename}")

    # Check contents for the files that do overlap.
    unique = set(v4) & set(v6)
    for filename in tqdm(unique, desc="Comparing Files"):
        v4 = os.path.join(jasa_dir, "v4", "flac", filename)
        v6 = os.path.join(jasa_dir, "v6", "flac", filename)

        hash4 = hashlib.md5(open(v4, "rb").read()).hexdigest()
        hash6 = hashlib.md5(open(v6, "rb").read()).hexdigest()
        if hash4 != hash6:
            print(f"Hash mismatch: {filename}")
        else:
            # For the files that do match exactly, copy their gt/machine result to the destination.
            transcribed_filename = filename.replace(".flac", ".json")
            source_fqn = os.path.join(
                jasa_dir, "v4", "gt", transcribed_filename
            )
            target_fqn = os.path.join(
                jasa_dir, "v6", "gt", transcribed_filename
            )
            shutil.copy(source_fqn, target_fqn)

            source_fqn = os.path.join(
                jasa_dir, "v4", "machine-video", transcribed_filename
            )
            target_fqn = os.path.join(
                jasa_dir, "v6", "machine-video", transcribed_filename
            )
            shutil.copy(source_fqn, target_fqn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
