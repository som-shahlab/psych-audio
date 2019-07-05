"""
Counts the number of scrubbed words across all ground truth files.
"""
import os
import sys
import json
import argparse
from tqdm import tqdm


def main(args):
    if not os.path.exists(args.gt_dir):
        print(f"Path does not exist: {args.gt_dir}")
        sys.exit(0)

    ls = os.listdir(args.gt_dir)
    scrubbed = 0
    total = 0
    for i in tqdm(range(len(ls))):
        filename = ls[i]
        # Load and standardize the ground truth.
        fqn = os.path.join(args.gt_dir, filename)
        with open(fqn, "r") as f:
            A = json.load(f)

        for B in A["results"]:
            for C in B["alternatives"]:
                for D in C["words"]:
                    word = D["word"]
                    if "[" in word and "]" in word:
                        scrubbed += 1
                    total += 1

    pct = 100 * scrubbed / total
    print(f"Scrubbed: {scrubbed} / {total} ({pct:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gt_dir",
        type=str,
        help="Location of the ground truth json transcription.",
    )
    main(parser.parse_args())
