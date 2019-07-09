"""
This file contains helper functions for the other pre-processing files.
"""
import os
import re
import string
from typing import *

# Used for converting numbers to words.
num2words1 = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
num2words2 = [
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
ordinals = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "5th": "fifth",
    "6th": "sixth",
    "7th": "seventh",
    "8th": "eighth",
    "9th": "ninth",
    "10th": "tenth",
}

# Remove these words, either because Speech API does not transcribe them,
# or they are considered stop words.
remove_words = [
    "um",
    "mm",
    "mmhmm",
    "mmmm",
    "uh",
    "uhhuh",
    "hmm",
    "umhmm",
    "huh",
    "mmmhmm",
    "aa",
    "gonna",
    "umm",
    "hmhmm",
    "uhuh",
    "mmm",
    "ah",
    "uhhmm",
    "ii",
    "uhhum",
    "mmhm",
    "hm",
    "ccaps",
    "gotta",
    "imim",
    "eh",
    "ugh",
    "gotcha",
    "hmmm",
    "un",
    "likei",
    "microphone",
    "justi",
    "donti",
    "da",
    "toi",
    "ma",
    "whwhat",
    "reallyi",
    "iii",
    "wasi",
    "hap",
]
remove_symbols = ["—", "’", "“"]


def canonicalize_word(word: str) -> str:
    """
	Canonicalizes a single word.
	- Converts numbers 0-99 into words.
	- Converts to lower-case.
	- Removes scrubbed data (denoted by brackets, e.g., [laugh])
	- Removes punctuation.
	Note: Canonicalized, scrubbed words will become the empty string.

	From: https://stackoverflow.com/questions/19504350/how-to-convert-numbers-to-words-in-python
	"""
    word = word.lower()

    # Remove scrubbed data. Needs to happen before punctuation removal.
    if "[" in word and "]" in word:
        return ""
        # word = re.sub('\[.*\]|\s-\s.*', '', word)

    word = word.replace("—", " ")

    # Remove punctuation and other symbols.
    word = word.translate(str.maketrans("", "", string.punctuation))
    for sym in remove_symbols:
        word = word.replace(sym, "")

    # Convert numbers to words.
    if word.isdigit():
        word = digits_to_words(word)
    if word in ordinals:
        word = ordinals[word]

    # Remove everything except letters.
    # clean_word = ''
    # for c in word:
    # 	if c.isalpha():
    # 		clean_word += c

    # Remove stop words.
    if word in remove_words:
        return ""

    return word


def canonicalize_sentence(text: str) -> str:
    """
	Canonicalizes a full sentence.
	:param text: Input text as a string.
	:return: Cleaned-up text.
	"""
    # Convert to lower-case.
    text = text.lower()

    # Merge multiple spaces.
    text = re.sub(" +", " ", text).strip()
    text = text.strip()

    # Process each word.
    tokens = text.split(" ")
    for i in range(len(tokens)):
        tokens[i] = canonicalize_word(tokens[i])
    text = " ".join(tokens)

    text = text.lower()

    return text


def digits_to_words(digits: str) -> str:
    """
	Converts a number, represented as digits, into English words.
	e.g. 12 -> twelve, 48 -> forty eight
	:param digits: String containing the number.
	:return: String of the words.
	"""
    integer = int(digits)
    if 0 <= integer < 19:
        return num2words1[integer]
    elif 20 <= integer <= 99:
        tens, below_ten = divmod(integer, 10)
        return num2words2[tens - 2] + " " + num2words1[below_ten]
    else:
        return ""


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
        print(f"File does not exist: {fqn}")
        return False

    # Open and check each line.
    with open(fqn, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # Skip the first line since it contains headers.
            if i == 0:
                continue

            # Check if line is valid.
            tokens = line.strip().split("\t")
            if len(tokens) != 36:
                print(f"Line {i+1} does not have 36 columns.")
                return False

            fqns = tokens[32].split(";")
            # It is possible for this row to have zero filenames.
            if len(fqns) == 1 and fqns[0] == "":
                continue
            # If this row has non-empty audio filenames.
            for fqn in fqns:
                if not os.path.exists(fqn):
                    print(f"Line {i+1} audio does not exist: {fqn}")
                    return False

    return True


def remove_extension(filename: str) -> str:
    """
	Removes the mp3 or wma extension from a string.

	:param filename: Filename to parse.
	:return: Filename but without the extension.
	"""
    for ext in [".wma", ".mp3", ".wav"]:
        filename = filename.replace(ext, "")
        filename = filename.replace(ext.upper(), "")
    return filename


def get_mmss_from_time(text: str) -> (int, int):
    """
	Returns the minutes and seconds from `[TIME: MM:SS]`.

	:param text: Text in the form `[TIME: MM:SS]`
	:return: Minutes and seconds as ints.
	"""
    matches = [m.span() for m in re.finditer("([0-9]){2}", text)]
    if len(matches) != 2:
        print(f"Malformed timestamp: {text}")
        return None
    minute = int(text[matches[0][0] : matches[0][1]])
    seconds = int(text[matches[1][0] : matches[1][1]])
    return minute, seconds


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
        (r"\[+TIME: \d+:[0-5][0-9]\]", len("[TIME: MM:SS]")),
        (r"\[+TIME: \d+:[0-5][0-9]\]: ", len("[TIME: MM:SS]:")),
        (r"\[+TIME \d+:[0-5][0-9]\]", len("[TIME MM:SS]:")),
        (r"\[+TIME \d+:[0-5][0-9]\]:", len("[TIME MM:SS]:")),
    ]

    meta = []
    for item in patterns:
        p, _ = item
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
                next_idx, _ = meta[i + 1]
                text = line[text_start:next_idx]

            item = (ts, text)
            subphrases.append(item)

    return subphrases
