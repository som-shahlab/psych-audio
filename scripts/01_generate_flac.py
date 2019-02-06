""""
This script converts our psychotherapy raw audio files (e.g., /share/pi/nigam/psych_audio/)
into a standardized .flac format. Some features of the standardized data:
- Single channel (mono)
- 44.1 kHz sampling rate
- .flac audio format

Prerequisites:
- FFmpeg static binary.
    FFmpeg contains various audio/visual encoding and decoding formats. To install:
        1. (Recommended) Download a static binary from: https://johnvansickle.com/ffmpeg/
        2. Compile from source: https://www.ffmpeg.org/
    Your FFmpeg binary can be entirely in user-space (i.e., you do not need sudo).
- Python 3.7 or higher

Input:
    This script assumes `INPUT_DIR` contains the structure:
    e.g. 020000/020200/020201/S1_020201_P1_07.25.16.WMA

Output:
    The output of this script will be placed in `OUTPUT_DIR` and will be a single folder of flac files only.
    e.g. S1_020201_P1_07.25.16.flac, S2_020201_P1_08.08.16.flac, ...
"""
import os
import string
import argparse
import multiprocessing
from typing import Tuple

# Location of the folders and subfolders which contain the audio files.
INPUT_DIR = '/share/pi/nigam/psych_audio/raw-audio'

# Location where to place the output files. Directory will be created if already exists.
OUTPUT_DIR = '/home/akhaque/output'

# Path to the ffmpeg static binary. That is, if we execute the string, it launches ffmpeg directly.
ffmpeg = '/home/akhaque/ffmpeg-4.1-64bit-static/ffmpeg'


def main(args: argparse.Namespace):
    """Handles high-level filename collection, directory creation, and thread management."""
    # Create the output directory, if it does not exist.
    print('Creating the output directory...')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    else:
        print('Output directory already exists.')

    # Recursively get list of all audio filenames and their desired output filename.
    workload = []
    print('Gathering input filenames...')
    for path, _, filenames in os.walk(INPUT_DIR):
        for filename in filenames:
            # Get the base filename without the audio extension.
            basename = '.'.join(filename.split('.')[:-1])
            source_fqn = os.path.join(INPUT_DIR, path, filename)  # fqn: Fully-qualified name.
            dest_fqn = os.path.join(OUTPUT_DIR, f'{basename}.flac')
            workload.append((source_fqn, dest_fqn))

    # Create multiple threads.
    print(f'Starting {args.n_threads} threads...')
    with multiprocessing.Pool(args.n_threads) as pool:
        pool.map(worker, workload)


def worker(item: Tuple[str, str]):
    """
    Converts a single audio file `source_fqn` into a flac file and saves it to `dest_fqn`.

    :param item: Tuple containing two strings: source_fqn and dest_fqn.
        source_fqn: Absolute filename of the source audio.
        dest_fqn: Absolute filename of the destination flac file.
    """
    source_fqn, dest_fqn = item
    # -i = input, -c:a = audio codec, -ac = # output channels, -ar = output sampling rate
    cmd_template = string.Template(f'{ffmpeg} -i $source -c:a flac -ac 1 -ar 44100 $dest')
    cmd = cmd_template.substitute(source=source_fqn, dest=dest_fqn)
    print(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default=1, type=int)
    args = parser.parse_args()
    main(args)
