"""
This script uploads a folder of files onto Google cloud.

Prerequisites:
- You must have a google cloud API key (json file).
- You must have a single, local folder containing multiple files. No nested folders allowed.
"""
import os
import sys
import argparse
from google.cloud import storage

# Full path to the Google API key.
# KEY = '/vol0/psych_audio/Google/keyAdam.json'
KEY = '../d757ae27e128.json'

# Upload the data to this Google cloud bucket.
BUCKET_NAME = "psych-audio-ahaque"

# Audio source directory. Contains data to upload to the cloud.
DATA_DIR = '/home/ahaque/Github/psych-audio/data'


def main():
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)

    fqns = []  # TODO.
    for filename in os.listdir(DATA_DIR):
        fqn = os.path.join(DATA_DIR, filename)
        blob = bucket.blob(filename)  # Create the target blob/file on the bucket.
        print(f'Uploading: {fqn}')
        blob.upload_from_filename(fqn) # Upload the local file into the target blob/file.
        print(f'Done.')

if __name__ == '__main__':
    # Google python API requires we set the OS environment variable.
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY
    main()
