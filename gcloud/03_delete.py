"""
This script deletes all files in the GCloud bucket.
"""
import os
import sys
import config
import argparse
from google.cloud import storage

def main(args):
    """
    Main entry loop. Sets up the Google API and loads an audio file.
    :param args: Argparse argument list.
    :return: None
    """
    if not args.confirm:
        print(f'You must use --confirm to verify you are okay with deleting files from: {args.bucket_name}.')
        sys.exit(0)
    # List all audio files in the bucket.
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(args.bucket_name)
    blobs = bucket.list_blobs()
    # `blobs` is a list of Google blob objects. We need to extract filenames.
    for b in blobs:
        b.delete()
        print(f'Deleted: {b.name}')


if __name__ == '__main__':
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.KEY
    parser = argparse.ArgumentParser()
    parser.add_argument('bucket_name', type=str, help='Delete files inside this bucket.')
    parser.add_argument('--confirm', action='store_true', help='You must include this flag to confirm you wish to delete the files.')
    args = parser.parse_args()
    main(args)
