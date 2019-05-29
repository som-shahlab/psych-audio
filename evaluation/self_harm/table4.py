"""
Computes Table 4 (self-harm metrics).
"""
import os
import sys
import json
import argparse
import pandas as pd
from typing import *
import preproc.util

# Location of the self-harm annotation file.
LABEL_FILE = '/vol0/psych_audio/gold-transcripts/self-harm-annotations.txt'

# Location of gold TXT with the hash filename.
TXT_DIR = '/vol0/psych_audio/gold-transcripts/gold-final_2019-04-08_anno'

# Location of the word-level GT.json and pred.json files.
JASA_DIR = '/vol0/psych_audio/jasa_format/v6'


def main(args):
    # Load the annotations into a nice format.
    # `examples` is a list of tuples, where each tuple contains
    # (hash, start char idx, end char idx) of the self-harm phrase.
    examples = load_annotations()

    # Generate paired sentences.
    paired = generate_paired(examples)


def generate_paired(examples: List) -> Dict:
    """
    Generates the paired sentences.
    This function will look at each annotation example and:
    1. Find the approximate location in the gold TXT files.
    2. Get the exact second/millisecond offsets using Google API results.
    3. Compose the paired GT/pred sentences.

    Args:
            examples (List): List of examples, where each example contains the
                    hash ID, the start character offset, and end char offset.

    Returns:
            Dict: Dictionary with key: example ID, value: gt/pred sentences.
    """
    result = {}
    for example in examples:
        hash_, start, end, phrase = example
        # For each example, load the gold TXT file.
        txt_fqn = os.path.join(TXT_DIR, f'{hash_}.txt')
        with open(txt_fqn, 'r') as f:
            data = f.read()

        # Get the start and end time for this example.
        start_ts, end_ts = get_start_end_ts(data, start, end)
        
        # Get the canonicalized GT and machine's prediction.
        gt, pred, confidences = get_paired_phrase(hash_, start_ts, end_ts)
        print(gt)
        print(pred)
        print(confidences)

    return result


def get_paired_phrase(hash_: str, start_ts: float, end_ts: float) -> Dict:
    """
    For a given start and end timestamp, gets the GT and pred phrase
    from the JSON file.

    Note: We cannot used `paired.json` because that file does not have
    word-level confidences or timestamps.
    
    Args:
        hash_ (str): Hash ID of the session.
        start_ts (float): Start time in seconds.
        end_ts (float): End time in seconds.
    
    Returns:
        gt (str): Ground truth sentence.
        pred (str): Predicted sentence.
        confidences (List[float]): Confidences for each pred word.
    """
    gt, pred = [], []
    confidences = []

    gt_fqn = os.path.join(JASA_DIR, 'gt', f'{hash_}.json')
    gt_items = get_words_between(gt_fqn, start_ts, end_ts)

    pred_fqn = os.path.join(JASA_DIR, 'machine-video', f'{hash_}.json')
    pred_items = get_words_between(pred_fqn, start_ts, end_ts)

    # Compose the final strings and confidence array.
    for item in gt_items:
        gt.append(item['word'])
    for item in pred_items:
        pred.append(item['word'])
        confidences.append(item['conf'])

    # Convert to string.
    gt = ' '.join(gt)
    pred = ' '.join(pred)

    return gt, pred, confidences


def get_words_between(json_fqn: str, start_ts: float, end_ts: float):
    """
    Loads a json transcription file and returns the words between `start` and
    `end` timestamps (in seconds).

    A = json.load(fqn)
    A['results'] is a Python list of dictionaries.
    B = A['results'][0]
    B['alternatives'] is a Python list of dictionaries.
    C = B['alternatives'][0] is a dictionary of transcription results.
        C Keys: transcript, confidence, words
    C['words'] is a Python list of dictionaries.
    D = C['words'][0] contains the transcription for a single word.
        D Keys: startTime, endTime, word, confidence, speakerTag

    Args:
        json_fqn (str): Path to the json file to load.
        start_ts (float): Start timestamp in seconds.
        end_ts (float): End timestamp in seconds.
    """
    with open(json_fqn, 'r') as f:
        A = json.load(f)

    words = []  # List of tuples, each tuple = word, confidence, speaker
    # For each word, add it to our list.
    for B in A['results']:
        for C in B['alternatives']:
            # Sometimes the 'words' key is not present.
            if 'words' not in C:
                continue
            for D in C['words']:
                # Get the core content.
                ts = float(D['startTime'].replace('s', ''))

                if ts < start_ts or ts > end_ts:
                    continue

                item = {}
                item['word'] = preproc.util.canonicalize_word(D['word'])
                item['start_ts'] = ts
                if 'speakerTag' in D.keys():
                    item['speaker_tag'] = D['speakerTag']
                if 'end_ts' in D.keys():
                    item['end_ts'] = D['end_ts']
                if 'confidence' in D.keys():
                    item['conf'] = D['confidence']

                words.append(item)

    return words


def get_start_end_ts(full_text: str, start: int, end: int) -> (float, float):
    """
    Gets the start and end timestamp for this entire line. 

    This function finds the starting point of this line and the HH:MM timestamp
    on the next line.

    Args:
        ful_text (str): Full body text to search (i.e., transcript).
        start (int): Character offset for the starting position.
        end (int): Character offset for the ending position.

    Returns:
        start_ts (float): Starting time in seconds.
        end_ts (float): Ending time in seconds.
    """
    # Find the start timestamp by finding the start and end of this line.
    start_of_line = full_text[:end].rfind('\n') + 1
    end_of_line = -1
    for i in range(end, len(full_text)):
        if ord(full_text[i]) == 10:
            end_of_line = i
            break
    
    result = full_text[start_of_line: end_of_line]
    start_ts_string = result[result.find('['):result.find(']')+1]
    start_min, start_sec = preproc.util.get_mmss_from_time(start_ts_string)
    start_ts = float(start_min * 60 + start_sec)

    # Find the starting time of the next line. This will be used as the
    # ending time of the first line.
    substart_idx, subend_idx = -1, -1
    for i in range(end, len(full_text)):
        if full_text[i] == '[':
            substart_idx = i
        elif full_text[i] == ']':
            subend_idx = i
            break
    
    end_ts_string = full_text[substart_idx:subend_idx]
    end_min, end_sec = preproc.util.get_mmss_from_time(end_ts_string)
    end_ts = float(end_min * 60 + end_sec)

    return start_ts, end_ts


def load_annotations() -> List[Tuple]:
    """
    Loads the annotation file.
    """
    df = pd.read_csv(LABEL_FILE, sep='\t', header=None,
                     names=['filename', 'offset', 'phrase'])
    examples = []
    # Parse each column.
    for _, row in df.iterrows():
        hash_ = row['filename'][:row['filename'].find('.ann:')]
        start, end = row['offset'].replace('Important ', '').split(' ')
        start, end = int(start), int(end)
        examples.append((hash_, start, end, row['phrase']))
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_embedding', action='store_true',
                        help='If True, does not compute embeddings.')
    main(parser.parse_args())
