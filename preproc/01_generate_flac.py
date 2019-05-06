# -*- coding: utf-8 -*-
"""
This script converts our psychotherapy raw audio files (e.g., /share/pi/nigam/psych_audio/)
into a standardized .flac format. Some features of the standardized data:
- Single channel (mono)
- 44.1 kHz sampling rate
- FLAC audio format: https://en.wikipedia.org/wiki/FLAC

How to run this code:
1. Fill in the correct values for `config.raw_audio_dir`, `args.output_dir` and `ffmpeg`.
2. Run this script.
    Option 1: Run on single machine (e.g., d01):
        $ python 01_generate_flac.py --n_threads N
        Note that ffmpeg inherently attempts to use all threads. We recommend running this script with 1 thread
        and monitoring CPU usage. If cores are under-utilized, then increase the thread count.
    Option 2: Run on the nero cluster. First, update `01_slurm.sh` with your requested number of CPUs and/or RAM.
        $ sbatch 01_slurm.sh
        To check on the status, run:
        $ squeue -u <your_sunetid>

Input:
    This script assumes `config.raw_audio_dir` contains the structure:
    e.g. 020000/020200/020201/S1_020201_P1_07.25.16.WMA

Output:
    The output of this script will be placed in `args.output_dir` and will be a single folder of flac files only.
    e.g. S1_020201_P1_07.25.16.flac, S2_020201_P1_08.08.16.flac, ...
"""
import os
import preproc.config
import string
import argparse
import subprocess
import multiprocessing
from typing import Tuple


def main(args: argparse.Namespace):
    """Handles high-level filename collection, directory creation, and thread management."""
    # Create the output directory, if it does not exist.
    print('Creating the output directory...')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print('Output directory already exists.')

    # Recursively get list of all audio filenames and their desired output filename.
    workload = []
    print('Gathering input filenames...')
    for path, _, filenames in os.walk(preproc.raw_audio_dir):
        for filename in filenames:
            # Get the base filename without the audio extension.
            basename = '.'.join(filename.split('.')[:-1])
            source_fqn = os.path.join(preproc.raw_audio_dir, path, filename)  # fqn: Fully-qualified name.
            dest_fqn = os.path.join(args.output_dir, f'{basename}.flac')
            # If we're continuing an interrupted job, we should skip any completed files.
            if args.resume:
                if os.path.exists(dest_fqn):
                    continue
            workload.append((source_fqn, dest_fqn))

    print(f'Found {len(workload)} files.')

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
    # Note that Google recommends 16,000 Hz. This is because their models are trained on 16,000.
    # Anything higher (e.g. 44kHz) significantly increases the filesize without any performance gains.
    cmd_template = string.Template(f'{preproc.03_create_gt_jsonconfig.ffmpeg} -i \"$source\" -c:a flac -ac 1 -ar 16000 \"$dest\"')
    cmd = cmd_template.substitute(source=source_fqn, dest=dest_fqn)

    # Execute on command line.
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str,
                        help='Location where to place the new flac files.')
    parser.add_argument('--n_threads', default=1, type=int,
                        help='Number of simultaneous FFmpeg calls. Note, all available threads are always used.')
    parser.add_argument('--resume', action='store_true',
                        help='If True, skip files already completed. Useful for resuming an interrupted job.')
    args = parser.parse_args()
    main(args)
