"""
This script converts the ground truth transcriptions from TXT format
into a structured JSON format, matching the Google Speech API format.
"""
import os
import re
import sys
import pprint
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from tqdm import tqdm
from pandas import DataFrame

meta_fqn = '/vol0/psych_audio/standard_data/metadata.tsv'
gt_dir = '/vol0/psych_audio/gold-transcripts/gold-final'


def main(args):
    # Ensure the metadata file is valid.
    if not args.no_meta_check:
        if not metadata_file_is_clean(meta_fqn):
            print(f'File contains errors: {meta_fqn}')
            return
        else:
            print('Metadata file OK.')

    meta = pd.read_csv(meta_fqn, sep='\t')

    # Create the audio filename to has mapping.
    audio2hash = create_audio2hash_map(meta)

    # Check if any GT files map to multiple audio files.
    convert_ground_truths_to_json(audio2hash, gt_dir)


def create_audio2hash_map(meta: DataFrame) -> Dict[str, str]:
    """
    Creates a dictionary with key=audio filename and value=hash.

    :param meta: Pandas DataFrame of the metadata file.
    :return: Dictionary which maps filenames to hash.
    """
    mapping = {}
    for i, row in meta.iterrows():
        if i == 0: continue  # Skip the header.
        audio_path_str = str(row['audio_path'])
        if audio_path_str == 'nan':
            continue

        paths = audio_path_str.split(';')  # Split, in case has multiple audio files.
        hash = row['hash']

        for path in paths:
            filename = os.path.basename(path)
            for ext in ['.wma', '.WMA', 'mp3', '.MP3']:
                filename = filename.replace(ext, '')
            mapping[filename] = hash

    return mapping


def convert_ground_truths_to_json(audio2hash: Dict[str, str], gt_dir: str):
    """
    Converts a folder containing TXT ground truth transcriptions into a folder of JSON files.

    :param audio2hash: Mapping from audio filename to hash.
    :param gt_dir: Directory containing TXT files.
    """
    trans_filenames = os.listdir(gt_dir)
    for filename in tqdm(trans_filenames):
        audio_filename, result = gt2dict(os.path.join(gt_dir, filename))
        hash = audio2hash[audio_filename]


def gt2dict(trans_fqn: str) -> (str, Dict):
    """
    Converts a ground truth human transcription file in the format:
        X [TIME: MM:SS] Transcribed sentence containing various words like this.
    where X is T=therapist or P=patient and MM:SS is the time.

    :param trans_fqn: Full path to the ground truth transcription.
    :return:
        Full path to the audio file for this transcription.
        Dictionary containing the transcriptions, in the same format as Google Speech API.
    """
    # Create the mapping of audio filename to hash.
    with open(trans_fqn, 'r') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]  # Remove newlines.

    audio_filename = None
    result = {}
    for line_no, line in enumerate(lines):
        # First four lines are header data.
        # First line is in the format: `Audio filename: XXX` where XXX is a variable-length audio filename.
        if line_no == 0:
            audio_filename = line.replace('Audio filename: ', '')
        elif line_no == 1 and 'Transcript filename:' not in line \
                or line_no == 2 and 'Therapist (T) gender:' not in line \
                or line_no == 3 and 'Patient (P) gender:' not in line:
            print(f'Malformed transcript file, line {line_no+1}: {trans_fqn}')
            return
        elif line_no == 4:
            continue
        elif line_no >= 5:
            # Extract the speaker ID and time.
            speaker_id = line[0]
            subphrases = get_subphrases(line)

            for phrase in subphrases:
                time_str, text = phrase
                mm, ss = get_mmss_from_time(time_str)
                ts = f'{mm * 60 + ss}.000s'
                words = text.split(' ')
                words_label = [{'startTime': ts, 'word': x} for x in words]

                label = {
                    'alternatives': {
                        'transcript': text,
                        'words': words_label,
                        'speakerTag': speaker_id,
                    }
                }
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(label)


    return audio_filename, result


def get_subphrases(line: str) -> List[Tuple[str, str]]:
    """
    Given a ground truth transcription, extracts all subphrases.
    A subphrase is defined as a set of words with a single timestamp.
    In our ground truth file, it is possible for a single line to contain multiple timestamps,
    corresponding to different phrases. This function extracts each individual phrase
    so we can create the json file with precise timestamps.

    :param line: Text from the transcription file.
    :return: List of subphrases, where each subphrase is defined by the timestamp and words.
    """
    # Find all timestamps on this line.
    # Finds: `[TIME: MM:SS]:` with or without the leading or ending colon.
    patterns = [
        ('\[+TIME: ([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]\] ', len('[TIME: MM:SS]')),
        ('\[+TIME: ([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]\]: ', len('[TIME: MM:SS]:')),
        ('\[+TIME ([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]\]', len('[TIME MM:SS]:')),
        ('\[+TIME ([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]\]:', len('[TIME MM:SS]:')),
    ]

    meta = []
    for item in patterns:
        p, size = item
        idxs = [m.span() for m in re.finditer(p, line)]
        meta += idxs
    meta = list(reversed(meta))

    # Only one phrase in this line.
    subphrases = []
    if len(meta) == 1:
        start, end = meta[0]
        ts, text = line[start:end], line[end:]
        item = (ts, text)
        subphrases.append(item)
    elif len(meta) > 1:
        # Extract the text for the subphrase.
        for i in range(len(meta)):
            start, end = meta[i]
            ts = line[start:end]
            text_start = end
            # If this is the last phrase.
            if i == len(meta) - 1:
                text = line[text_start:]
            else:
                next_idx, next_size = meta[i + 1]
                text = line[text_start:next_idx]

            item = (ts, text)
            subphrases.append(item)

    return subphrases


def get_mmss_from_time(text: str) -> (int, int):
    """
    Returns the minutes and seconds from `[TIME: MM:SS]`.

    :param text: Text in the form `[TIME: MM:SS]`
    :return: Minutes and seconds as ints.
    """
    matches = [m.span() for m in re.finditer('([0-9]){2}', text)]
    if len(matches) != 2:
        print(f'Malformed timestamp: {text}')
        return None
    minute = int(text[matches[0][0]:matches[0][1]])
    seconds = int(text[matches[1][0]:matches[1][1]])
    return minute, seconds


def clean_up_transcript(text: str):
    pass


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
            # It is possible for this row to have zero filenames.
            if len(fqns) == 1 and fqns[0] == '':
                continue
            # If this row has non-empty audio filenames.
            for fqn in fqns:
                if not os.path.exists(fqn):
                    print(f'Line {i+1} audio does not exist: {fqn}')
                    return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_meta_check', action='store_true', help='Used for code development only.')
    args = parser.parse_args()
    main(args)
