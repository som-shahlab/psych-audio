"""
This script takes existing flac files and standardizes them according
to our metadata tsv file. This includes joining multiple flac files into
a single audio file (e.g., a single therapy session was split into two audio files).

Make sure to have finsihed running `preproc/01_generate_flac.py` before running this script.
"""
import os
import shutil
import argparse
import pandas as pd
import util
import config


def main(args):
    meta = pd.read_csv(config.meta_fqn, sep='\t')

    # For each metadata row, copy and rename the audio file.
    for i, row in meta.iterrows():
        hash = row['hash']
        path = str(row['audio_path'])

        if ' ' in path:
            print('Found space in file')

        if path == 'nan':
            continue
        elif ';' in path:
            # We have 2 paths. Need to concat.
            # TODO: Need to implement.
            pass
        else:
            filename = util.remove_extension(os.path.basename(path))
            if filename in config.malformed_files:
                print(f'Skipping malformed: {filename}')
                continue
            source_fqn = os.path.join(args.input_dir, f'{filename}.flac')
            dest_fqn = os.path.join(args.output_dir, f'{hash}.flac')
            if os.path.exists(dest_fqn):
                continue
            shutil.copy(source_fqn, dest_fqn)
            print(source_fqn, dest_fqn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str,
                        help='Directory which contains all flac files.')
    parser.add_argument('output_dir', type=str,
                        help='Location to place the new flac files.')
    args = parser.parse_args()
    main(args)
