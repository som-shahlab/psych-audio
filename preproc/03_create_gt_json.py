"""
This script converts the ground truth transcriptions from TXT format
into a structured JSON format, matching the Google Speech API format.
"""
import os
import re
import json
import argparse
import pandas as pd
from typing import Dict, Tuple, List
from tqdm import tqdm
from pandas import DataFrame
import preproc.util
import preproc.config
import evaluation.config


def main(args):
    # Load the metadata file.
    meta = pd.read_csv(preproc.config.meta_fqn, sep="\t")

    # For each test-set row, check if the gold and audio file exists.
    count = 0
    for i, row in meta.iterrows():
        asr_test = row["asr_test"]
        gold_path = row["gold_path"]
        hash_ = row["hash"]
        if asr_test:
            count += 1
            _, results = gt2dict(gold_path)
            # Confirm if audio file exists.
            flac_fqn = os.path.join(
                "/vol0/psych_audio/jasa_format/flac", f"{hash_}.flac"
            )
            if not os.path.exists(flac_fqn):
                print(f"Audio does not exist: flac_fqn")
                continue

            # Write the GT json file.
            out_fqn = os.path.join(args.output_dir, f"{hash_}.json")
            with open(out_fqn, "w") as f:
                json.dump(results, f, indent=2, separators=(",", ": "))


def gt2dict(trans_fqn: str) -> (List[str], Dict):
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
    with open(trans_fqn, "r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]  # Remove newlines.

    audio_filenames = None
    results = []
    for line_no, line in enumerate(lines):
        # First four lines are header data.
        # First line is in the format: `Audio filename: XXX` where XXX is a variable-length audio filename.
        if line_no == 0 and "audio" in line.lower():
            # Find start index of the filename.
            idx = line.find(":") + 1
            stripped_line = line[idx:].strip()
            if " " in stripped_line:
                audio_filenames = stripped_line.split(" ")
            else:
                audio_filenames = [stripped_line]
        elif "[TIME:" in line:
            # Extract the speaker ID and time.
            speaker_id = line[0].upper()
            subphrases = preproc.util.get_subphrases(line)

            for phrase in subphrases:
                time_str, text = phrase
                mm, ss = preproc.util.get_mmss_from_time(time_str)
                ts = f"{mm * 60 + ss}.000s"
                words = []
                for x in text.split(" "):
                    if len(x) > 0:
                        words.append(x)

                # Compose the JSON entries.
                words_label = [
                    {"startTime": ts, "word": x, "speakerTag": speaker_id}
                    for x in words
                ]
                label = {
                    "alternatives": [
                        {"transcript": text, "words": words_label}
                    ],
                    "languageCode": "en-us",
                }
                results.append(label)

    results = {"results": results}
    return audio_filenames, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", type=str, help="Location to store the output JSON files."
    )
    parser.add_argument(
        "--no_meta_check",
        action="store_true",
        help="Used for code development only.",
    )
    args = parser.parse_args()
    main(args)
