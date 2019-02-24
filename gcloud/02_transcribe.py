"""
This script opens a single audio file, sets up the Google API,
uploads the audio bytes to Google, and returns the
diarized (speaker-separated) transcription.
"""
import os
import config
import argparse
import numpy as np
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import SpeechClient
from google.cloud.speech_v1p1beta1.types import RecognitionConfig, RecognitionAudio


def main(args):
    """
    Main entry loop. Sets up the Google API and loads an audio file.
    :param args: Argparse argument list.
    :return: None
    """
    # Create a single Google API client and configuration to reuse.
    # For a list of configuration options, see the Google Speech API documentation:
    # https://cloud.google.com/speech-to-text/docs/word-confidence
    client = speech.SpeechClient()

    config = RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',
        enable_speaker_diarization=True,
        enable_word_confidence=True,
        enable_word_time_offsets=True,
        diarization_speaker_count=2,
    )

    # Load the audio file.
    speech_file = '../audio16khz.wav'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio_bytes = RecognitionAudio(content=content)

    ret = transcribe(client, config, audio_bytes)


def transcribe(client: SpeechClient, config: RecognitionConfig, audio_bytes: bytes):
    """
    Makes the API call to transcribe `audio_bytes`.

    :param client: Google API speech client.
    :param config: Google API object containing the language, sample rate, etc.
    :param audio_bytes: Bytes object, of length (audio_length_in_sec * 2 * sample rate).
    :return: transcription: List of tuples (word, speaker id), ordered by when each word occurs.
    """
    print('Waiting for operation to complete...')
    operation = client.long_running_recognize(config, audio_bytes)
    result = operation.result(timeout=90)

    # print('Waiting for operation to complete...')
    # response = client.recognize(config, audio_bytes)
    # # The transcript within each result is separate and sequential per result.
    # # However, the words list within an alternative includes all the words
    # # from all the results thus far. Thus, to get all the words with speaker
    # # tags, you only have to take the words list from the last result:
    # result = response.results[-1]
    #
    # words_info = result.alternatives[0].words
    # transcript = result.alternativesre

    for result in result.results:
        alternative = result.alternatives[0]
        print(u'Transcript: {}'.format(alternative.transcript))
        print('Confidence: {}'.format(alternative.confidence))

        # For each word, print the results.
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            confidence = word_info.confidence
            print('Word: {}, conf: {}, start_time: {}, end_time: {}'.format(
                word,
                confidence,
                start_time.seconds + start_time.nanos * 1e-9,
                end_time.seconds + end_time.nanos * 1e-9))

    return result


if __name__ == '__main__':
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.key
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
