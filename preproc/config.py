"""
This file contains various global constants used for data preprocessing.
"""

# The metadata file contains labels and patient information for each audio file.
meta_fqn: str = '/vol0/psych_audio/jasa_format/metadata.tsv'

# The ground truth directory contains TXT file transcriptions of a subset of the audio files.
gt_dir: str = '/vol0/psych_audio/gold-transcripts/gold-final'

# Location of the folders and subfolders which contain the audio files.
raw_audio_dir: str = '/vol0/psych_audio/raw-audio'

# Path to the ffmpeg static binary. That is, if we execute the string, it launches ffmpeg directly.
#    FFmpeg contains various audio/visual encoding and decoding formats. To install:
#        1. (Recommended) Download a static binary and place in somewhere: https://johnvansickle.com/ffmpeg/
#        2. Compile from source: https://www.ffmpeg.org/
#    Your FFmpeg binary can be entirely in user-space (i.e., you do not need sudo).
ffmpeg = '/vol0/psych_audio/ahaque/ffmpeg/ffmpeg'
