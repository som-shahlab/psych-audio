"""
This script uploads a folder of files onto Google cloud.

Prerequisites:
- You must have a google cloud API key (json file).
- You must have a single, local folder containing multiple files. No nested folders allowed.
"""
import os
import sys
import argparse
import config
from tqdm import tqdm
from google.cloud import storage

# Audio source directory. Contains data to upload to the cloud.
DATA_DIR = '../data'


def main():
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(config.BUCKET_NAME)

    print(f'Uploading files from: {DATA_DIR}')
    for filename in tqdm(os.listdir(DATA_DIR)):
        fqn = os.path.join(DATA_DIR, filename)
        blob = bucket.blob(filename)  # Create the target blob/file on the bucket.
        blob.upload_from_filename(fqn) # Upload the local file into the target blob/file.

if __name__ == '__main__':
    # Google python API requires we set the OS environment variable.
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.KEY
    main()
