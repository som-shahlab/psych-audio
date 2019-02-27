"""
This script processes audio files which already reside in a Google bucket.
It will loop over all files, transcribe them, and write the json transcription result
locally, to the machine running this script.
"""
import os
import json
import config
import argparse
from tqdm import tqdm
from google.cloud import storage
from google.cloud import speech_v1p1beta1 as speech
from google.protobuf.json_format import MessageToDict
from google.cloud.speech_v1p1beta1 import SpeechClient
from google.cloud.speech_v1p1beta1.types import RecognitionConfig, RecognitionAudio


def main(args):
    """
    Main entry loop. Sets up the Google API and loads an audio file.
    :param args: Argparse argument list.
    :return: None
    """
    # Check if the output directory exists.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # List all audio files in the bucket.
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(config.BUCKET_NAME)
    blobs = bucket.list_blobs()
    # `blobs` is a list of Google blob objects. We need to extract filenames.
    filenames = [b.name for b in blobs]

    # Create a single Google API client and configuration to reuse.
    # For a list of configuration options, see the Google Speech API documentation:
    # https://cloud.google.com/speech-to-text/docs/word-confidence
    client = speech.SpeechClient()
    rc = RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code='en-US',
        enable_speaker_diarization=True,
        enable_word_confidence=True,
        enable_word_time_offsets=True,
        diarization_speaker_count=2,
    )

    print(f'Saving json output to: {args.output_dir}')
    print(f'Transcribing {len(filenames)} files from bucket: {config.BUCKET_NAME}')
    for filename in tqdm(filenames):
        # Run ASR.
        audio = RecognitionAudio(uri=f'gs://{config.BUCKET_NAME}/{filename}')
        ret = transcribe(client, rc, audio)

        # Save the output to json.
        output_fqn = os.path.join(args.output_dir, filename.replace('.flac', '.json'))
        with open(output_fqn, 'w') as pointer:
            json.dump(ret, pointer, indent=2, separators=(',', ': '))


def transcribe(client: SpeechClient, rc: RecognitionConfig, audio: RecognitionAudio):
    """
    Makes the API call to transcribe `audio`.

    :param client: Google API speech client.
    :param rc: Google API object containing the language, sample rate, etc.
    :param audio: Google RecognitionAudio object. This refers to a blob in a google storage bucket.
    :return result: Dictionary of transcription results.
    """
    operation = client.long_running_recognize(rc, audio)
    response = operation.result()
    result = MessageToDict(response)
    return result


if __name__ == '__main__':
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.KEY
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, help='Location for the transcription output.')
    args = parser.parse_args()
    main(args)
