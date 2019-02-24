# Psychotherapy Analysis Project

![Banner Image](doc/banner.png)

## 1 Introduction

The aim of this project is to investigate whether automatic speech recognition can replace
the manual transcription and coding process, with the hope of accelerating psychotherapy research.
We collected a dataset containing 800 hours from psychotherapy sessions.
We perform automatic transcription using publicly available cloud-based transcription
services and analyze their performance compared to gold standard
transcriptions by humans. Our analysis includes techniques from natural language processing.

## 2 Speech-to-Text with Google Cloud

### 2.1 Prerequisites
 [FFmpeg](https://www.ffmpeg.org/) contains various audio/visual
encoding and decoding formats. To install FFmpeg:

  1. (Recommended) Download a static binary and place in home dir: https://johnvansickle.com/ffmpeg/
  2. Compile from source: https://www.ffmpeg.org/
  
Your FFmpeg binary can be entirely in user-space (i.e., you do not need sudo).


### 2.2 Generating FLAC Files

In its raw form, our current audio files are in either WMA (windows media audio) or MP3
format. As [recommended by Google Cloud](https://cloud.google.com/speech-to-text/docs/best-practices),
we convert our files to [FLAC](https://en.wikipedia.org/wiki/FLAC). In general, you should try
to use FLAC for all your audio processing tasks. The MP3 format loses data during the compression
process. While this is okay for human hearing (MP3 minimizes human perceptual data loss),
it may lose important information for machine hearing tasks.

Below, we show the original (mp3/wma) specs of our data and the specs of our new FLAC files. 
Most of these settings are [recommended](https://cloud.google.com/speech-to-text/docs/best-practices) by Google.
* Format: MP3/WMA -> FLAC
* Sample Rate: 44,100 Hz -> 16,000 Hz
* Channels: 2 (stereo) -> 1 (mono)

First, open [preprocessing/01_generate_flac.py](preprocessing/01_generate_flac.py)
and edit the global variables: `INPUT_DIR`, `OUTPUT_DIR`, and `ffmpeg`. Then, run:

```bash
python preprocessing/01_generate_flac.py
```

If running on NERO or other compute cluster, submit your job with
Slurm (see [preprocessing/01_slurm.sh](preprocessing/01_slurm.sh)).:

```bash
sbatch scripts/01_slurm.sh
squeue
```

The output flac files will be placed in `OUTPUT_DIR`.

### 2.3 Upload to Google Cloud

Once the audio files have been cleaned and standardized, we now upload the files to Google Cloud. It is better to upload the files and have them stored on GCloud to avoid re-uploading in case we need to re-run ASR.

### 2.4 Speech-to-Text

Todo: Run ASR on the flac files from the bucket.
