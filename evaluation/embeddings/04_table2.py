"""
This file creates a Table 2 (quant results) given a single CSV file of GID,metric pairs.

Prerequisites:
- Result csv files.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import evaluation.util
from evaluation import config
from typing import Dict, List, Set, Any


def main(args):
    """
	Main function which handles program flow, etc.
	:param args: Argparse namespace.
	"""
    if not os.path.exists(config.META_FQN):
        print(f"File does not exist: {config.META_FQN}")
        sys.exit(1)

    # Load the paired file. We want ALL the data.
    paired = evaluation.util.load_paired_json(skip_empty=False)

    # Loop through all CSV files and organize the results by hash as the primary key.
    hash2metrics = sort_csv_by_hash(args, paired)

    # For each hash, determine the values for each dimension of interest.
    hash2dim_values, unique_dim_vals = load_dimensions()

    # Create the tree-dictionary of metadata dimensions.
    # Key Structure: embedding_name, dimension, dim_value
    # Values: List of distances (float).
    accumulator: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    for embedding_name in config.F.keys():
        # Create the top level keys for embedding names.
        accumulator[embedding_name]: Dict[str, Dict[int, List[float]]] = {}
        for dim in unique_dim_vals.keys():
            # Create the keys for dimension names.
            accumulator[embedding_name][dim]: Dict[int, List[float]] = {}
            for val in unique_dim_vals[dim]:
                accumulator[embedding_name][dim][val]: List[float] = []

    # Populate the accumulator with our hash2metrics data.
    for hash in hash2metrics:
        # For each dimension, find the dimension value for this hash file.
        for dim in config.DIMENSIONS:
            if dim not in hash2dim_values[hash]:
                continue
            dim_val_for_this_hash = hash2dim_values[hash][dim]
            for embedding_name in config.F.keys():
                dists = hash2metrics[hash][embedding_name]
                accumulator[embedding_name][dim][dim_val_for_this_hash] += dists

    # For each dimension and unique value, print the mean.
    out_dir = os.path.join(config.DISTANCES_DIR, args.distance)
    results_fqn = os.path.join(out_dir, "table2.csv")
    with open(results_fqn, "w") as f:
        header = "dim,val"
        for name in config.F.keys():
            header += f",{name}_n,{name}_mean,{name}_std"
        f.write(header + "\n")

        for dim in unique_dim_vals.keys():
            for val in unique_dim_vals[dim]:
                result = f"{dim},{val}"
                for name in config.F.keys():
                    mean, std, n = get_mean_std(name, accumulator, dim, val)
                    if mean is None:
                        result += ",,,"
                    else:
                        result += f",{n},{mean},{std}"
                f.write(result + "\n")
    print(results_fqn)


def get_mean_std(embedding_name: str, accumulator: Dict, dim: str, val: int):
    """
	Checks if the embedding name, dim, val combo is empty or not in `accumulator`.
	:param embedding_name: bert, glove, or word2vec
	:param accumulator: Accumulator dictionary of distance values.
	:param dim: Dimension name.
	:param val: Dimension value.
	:return mean: Mean value of the array.
	:return std: Standard deviation.
	:return n: Number of elements.
	"""
    values = np.asarray(accumulator[embedding_name][dim][val])
    n = len(values)
    if n == 0:
        return None, None, None
    mean = values.mean()
    std = values.std()
    return mean, std, n


def load_dimensions() -> (Dict[str, Dict[str, int]], Dict[str, List[int]]):
    """
	Loads the metadata file and returns various dimensions for each hash file.
	:return hash2dims: Dictionary with key=hash, value=Dict of dimensions.
	:return unique_dim_vals: For each dimension, contains the unique values.
	"""
    hash2dims = {}
    unique_dim_vals = {x: set() for x in config.DIMENSIONS}
    df = pd.read_csv(config.META_FQN, sep="\t")
    for _, row in df.iterrows():
        hash_ = row["hash"]
        # Skip hashes already seen.
        if hash_ in hash2dims:
            print(f"Duplicate: {hash_}")
            continue
        # Populate the dimension's values.
        values = {}
        for key in config.DIMENSIONS:
            v = row[key]
            if str(v) == "nan":
                continue
            else:
                v = int(v)
                values[key] = v
                unique_dim_vals[key].add(v)
        hash2dims[hash_] = values

    return hash2dims, unique_dim_vals


def sort_csv_by_hash(args, paired: Dict) -> Dict[str, Dict[str, List[float]]]:
    """
	For each hash, collects all distances for that hash. Note that a single hash
	may consist of multiple GIDs.

	:param args: Argparse namespace.
	:param paired: Paired data from the json file.
	:return hash2metrics: Dictionary with key: hash, and value: dictionary of embedding distances.
	"""
    hash2metrics = {}
    # For each distance file, populate the accumulators based on the gid's hash value.
    for embedding_name in config.F.keys():
        out_dir = os.path.join(config.DISTANCES_DIR, args.distance)
        csv_fqn = os.path.join(out_dir, f"{embedding_name}.csv")
        df = pd.read_csv(csv_fqn, sep=",")

        # Loop over each row in the CSV distance file and add to the accumulator.
        for i, row in df.iterrows():
            # Get the data.
            gid, value = str(int(row["gid"])), row["value"]
            hash = paired[gid]["hash"]

            # If new hash, populate the dict with accumulator lists.
            if hash not in hash2metrics:
                hash2metrics[hash] = {k: [] for k in config.F.keys()}

            # Add to accumulator.
            hash2metrics[hash][embedding_name].append(value)

    return hash2metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distance", type=str, choices=["cosine", "wasserstein"]
    )
    main(parser.parse_args())
