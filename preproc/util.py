"""
This file contains helper functions for the other pre-processing files.
"""
import os


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


def remove_extension(filename: str) -> str:
    """
    Removes the mp3 or wma extension from a string.

    :param filename: Filename to parse.
    :return: Filename but without the extension.
    """
    for ext in ['.wma', '.mp3', '.wav']:
        filename = filename.replace(ext, '')
        filename = filename.replace(ext.upper(), '')
    return filename
