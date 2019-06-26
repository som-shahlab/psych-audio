"""
Runs the command-line based annotation tool.
"""
import os
import sys
import json
import argparse
import pandas as pd
from typing import *
from collections import Counter
import preproc.util
from evaluation.self_harm import config


def main(args):
    # Load the annotations into a nice format.
    # `examples` is a list of tuples, where each tuple contains
    # (hash, start char idx, end char idx) of the self-harm phrase.
    examples = load_annotations()

    # Generate paired sentences.
    paired = generate_paired(examples)

    # For each example, ask the user to select the start and end word
    # from the predicted text.
    for gid in paired.keys():
        if gid < args.start:
            continue
        item = paired[gid]
        print("-" * 80)
        gt = preproc.util.canonicalize_sentence(item["phrase"])
        pred = item["pred"]
        tokens = pred.split(" ")
        for i, word in enumerate(tokens):
            print(f"{i}\t{word}")

        print(f"GT ({gid}): {gt}")
        start = input("Start index? ")
        end = input("End index? ")
        if start != "" and end != "":
            start = int(start)
            end = int(end)
            subset_words = tokens[start : end + 1]
            subset_conf = item["conf"][start : end + 1]
        else:
            subset_words = [" "]
            subset_conf = [1.0]

        # Create the pandas dataframe.
        df_gt = []
        df_pred = []
        df_conf = []
        gt_words = gt.split(" ")
        N = max(len(subset_words), len(gt_words))
        for i in range(N):
            if i < len(gt_words):
                clean = preproc.util.canonicalize_word(gt_words[i])
                df_gt.append(clean)
            else:
                df_gt.append("")

            if i < len(subset_words):
                df_pred.append(subset_words[i])
                df_conf.append(subset_conf[i])
            else:
                df_pred.append("")
                df_conf.append(1.0)

        df = pd.DataFrame(data={"gt": df_gt, "pred": df_pred, "conf": df_conf})
        print(df)

        # Write to file.
        out_fqn = os.path.join(config.OUT_DIR, f"{str(gid).zfill(3)}.tsv")
        df.to_csv(out_fqn, sep="\t", index_label="wid")
        print(out_fqn)


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
    eid = 1
    for example in examples:
        hash_, start, end, phrase = example
        # For each example, load the gold TXT file.
        txt_fqn = os.path.join(config.TXT_DIR, f"{hash_}.txt")
        with open(txt_fqn, "r") as f:
            data = f.read()

        # Get the start and end time for this example.
        start_ts, end_ts = get_start_end_ts(data, start, end)

        if start_ts == -1:
            continue

        # Get the canonicalized GT and machine's prediction.
        gt, pred, confidences, speaker = get_paired_phrase(
            hash_, start_ts, end_ts
        )
        print(speaker)

        result[eid] = {
            "gt": gt,
            "pred": pred,
            "conf": confidences,
            "phrase": phrase,
        }
        eid += 1

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

    speaker = ""
    gt_fqn = os.path.join(config.JASA_DIR, "gt", f"{hash_}.json")
    gt_items, speaker = get_words_between(gt_fqn, start_ts, end_ts)

    pred_fqn = os.path.join(config.JASA_DIR, "machine-video", f"{hash_}.json")
    pred_items, _ = get_words_between(pred_fqn, start_ts, end_ts, is_pred=True)

    # Compose the final strings and confidence array.
    speakers = []
    for item in gt_items:
        gt.append(item["word"])
        speakers.append(item["speaker_tag"])
    for item in pred_items:
        pred.append(item["word"])
        confidences.append(item["conf"])

    # Convert to string.
    gt = " ".join(gt)
    pred = " ".join(pred)

    return gt, pred, confidences, speaker


def get_words_between(json_fqn: str, start_ts, end_ts, is_pred=False):
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
    pad_start_ts = start_ts - 10 if is_pred else start_ts
    pad_end_ts = end_ts + 10 if is_pred else end_ts
    with open(json_fqn, "r") as f:
        A = json.load(f)

    words = []  # List of tuples, each tuple = word, confidence, speaker
    speakers = Counter()
    # For each word, add it to our list.
    for B in A["results"]:
        for C in B["alternatives"]:
            # Sometimes the 'words' key is not present.
            if "words" not in C:
                continue
            for D in C["words"]:
                # Get the core content.
                ts = float(D["startTime"].replace("s", ""))

                if ts < pad_start_ts or ts > pad_end_ts:
                    continue

                item = {"word": preproc.util.canonicalize_word(D["word"])}
                item["start_ts"] = ts

                if "speakerTag" in D.keys():
                    item["speaker_tag"] = D["speakerTag"]
                    if not is_pred and ts >= start_ts and ts < end_ts:
                        speakers[D["speakerTag"]] += 1
                if "end_ts" in D.keys():
                    item["end_ts"] = D["end_ts"]
                if "confidence" in D.keys():
                    item["conf"] = D["confidence"]

                words.append(item)

    speaker = None
    if not is_pred:
        if sum(speakers.values()) == 0:
            print(words)
        else:
            speaker = speakers.most_common(1)[0]

    return words, speaker


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
    start_of_line = full_text[:end].rfind("\n") + 1
    end_of_line = -1
    for i in range(end, len(full_text)):
        if ord(full_text[i]) == 10:
            end_of_line = i
            break

    result = full_text[start_of_line:end_of_line]
    start_ts_string = result[result.find("[") : result.find("]") + 1]
    start_min, start_sec = preproc.util.get_mmss_from_time(start_ts_string)
    start_ts = float(start_min * 60 + start_sec)

    # Get a list of locations of the [TIME: HH:MM] string.
    subphrase = preproc.util.get_subphrases(full_text[end_of_line:])

    # Sometimes regex will fail. Therefore we have to manually check.
    if len(subphrase) == 0:
        pointer = -1
        W = 4
        # Find the location of TIME.
        for i in range(end_of_line, len(full_text) - W):
            window = full_text[i : i + W]
            if window == "TIME":
                pointer = i
                break

        # Find the precise location of the brackets.
        context = full_text[pointer - 1 : pointer + 13]
        bracket1 = context.find("[")
        bracket2 = context.find("]")

        end_time_str = context[bracket1 : bracket2 + 1]
    else:
        # Get the first result (need to reverse it).
        end_time_str, _ = subphrase[-1]

    end_min, end_sec = preproc.util.get_mmss_from_time(end_time_str)
    end_ts = float(end_min * 60 + end_sec)

    return start_ts, end_ts


def load_annotations() -> List[Tuple]:
    """
    Loads the annotation file.
    """
    df = pd.read_csv(
        config.LABEL_FILE,
        sep="\t",
        header=None,
        names=["filename", "offset", "phrase"],
    )
    examples = []
    # Parse each column.
    for _, row in df.iterrows():
        hash_ = row["filename"][: row["filename"].find(".ann:")]
        start, end = row["offset"].replace("Important ", "").split(" ")
        start, end = int(start), int(end)
        examples.append((hash_, start, end, row["phrase"]))
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", default=1, type=int, help="Example to start from."
    )
    main(parser.parse_args())
