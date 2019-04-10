# Psychotherapy Analysis Project

![Banner Image](doc/banner.png)

## 1 Introduction

The aim of this project is to investigate whether automatic speech recognition can replace
the manual transcription and coding process, with the hope of accelerating psychotherapy research.
We collected a dataset containing 800 hours from psychotherapy sessions.
We perform automatic transcription using publicly available cloud-based transcription
services and analyze their performance compared to gold standard
transcriptions by humans. Our analysis includes techniques from natural language processing.

## 2 Data Pre-Processing

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

### 2.3 Ground Truth JSON

The ground truth files are currently in TXT files, as delivered by the annotators.
We need to convert these to JSON as a standardization step.

1. Ensure that `gt_dir` is correct inside `preproc/config.py`.
2. Run the following inside `preproc/`.

```bash
python 03_create_gt_json.py OUTPUT_DIR
```

where `OUTPUT_DIR` is the target location to place the new, ground truth JSON files.

## 3 Speech-to-Text with Google Cloud

### 3.1 Prerequisites
First, enter your GCloud key and bucket information in [gcloud/config.py](gcloud/config.py).

- **Google Cloud API Key**. This should be a *service account*. The key should have permissions to Google Cloud storage and Google Speech-to-Text API. Easiest but not-security-recommended solution would grant the service account with *Project Owner* status/permissions.
- **Bucket Name**. This bucket should already exist. Our script will not create a bucket. The files will be uploaded to this bucket.

Second, install the python dependencies.
```bash
pip install -r requirements.txt
```

### 3.2 Upload to Google Cloud

Once the audio files have been cleaned and standardized, we now upload the files to Google Cloud. We have two options:
1. We send the audio file to Google for each transcription request. Nothing will reside long-term on GCloud.
2. Before transcription, we upload all audio files to GCloud as a one-time operation. These files may sit there for a while. The transcription script will point to these files.

Option 2 is better than Option 1. Specifically, when we wish to tweak our transcription algorithm or model. With option 2, we do not need to continually re-upload (temporarily) each audio file.

To upload the files, see [gcloud/01_upload.py](gcloud/01_upload.py). Run it with the following command:
```bash
cd gcloud
python 01_upload.py DATA_DIR
```

where `DATA_DIR` is the folder containing audio files. While running, the script will print the total upload progress.

### 3.3 Speech-to-Text

Now that we have a single bucket containing only flac files, we can run transcription and diarization.

```bash
cd gcloud
python 02_transcribe.py OUTPUT_DIR
```

where `OUTPUT_DIR` is your desired *local* folder where to store the json transcription results. This script will also print the transcription progress.

### 3.4 Compute Metrics

The final step is to compute metrics between the Google Speech API and the ground truth.
At this point, we should have two directories (and what we named them):

1. `machine`: Contains the Google Speech API output transcriptions as JSON format.
2. `gt`: Contains the ground truth transcriptions as JSON format (see Section 2.3).

To compute the metrics, run:

```bash
cd evaluation
python evaluate.py MACHINE_DIR GT_DIR
python comptue_metrics.py
```

The first script, `evaluate.py` computes metrics at a phrase-level. It also stores metadata such as the phrase start timestamp, speaker, and filename hash. This will procude `results.csv` which usually contains 10,000+ lines.

The second script, `compute_metrics.py` takes the `results.csv` file and prints out the final WER, BLEU, etc. metrics, as well as displays any relevant plots.