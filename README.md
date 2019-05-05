# Psychotherapy Analysis Project

![Banner Image](doc/banner.png)

## 1   Introduction

The aim of this project is to investigate whether automatic speech recognition can replace the manual transcription and coding process, with the hope of accelerating psychotherapy research. We collected a dataset containing 800 hours from psychotherapy sessions. We perform automatic transcription using publicly available cloud-based transcription services and analyze their performance compared to gold standard transcriptions by humans. Our analysis includes techniques from natural language processing.

## 2   Data Pre-Processing

### 2.1   Prerequisites
 [FFmpeg](https://www.ffmpeg.org/) contains various audio/visual encoding and decoding formats. To install FFmpeg:

  1. (Recommended) Download a static binary and place in home dir: https://johnvansickle.com/ffmpeg/
  2. Compile from source: https://www.ffmpeg.org/

Your FFmpeg binary can be entirely in user-space (i.e., you do not need sudo).


### 2.2   Generating FLAC Files

In its raw form, our current audio files are in either WMA (windows media audio) or MP3 format. As [recommended by Google Cloud](https://cloud.google.com/speech-to-text/docs/best-practices), we convert our files to [FLAC](https://en.wikipedia.org/wiki/FLAC). In general, you should try to use FLAC for all your audio processing tasks. The MP3 format loses data during the compression process. While this is okay for human hearing (MP3 minimizes human perceptual data loss), it may lose important information for machine hearing tasks.

Below, we show the original (mp3/wma) specs of our data and the specs of our new FLAC files.  Most of these settings are [recommended](https://cloud.google.com/speech-to-text/docs/best-practices) by Google.
* Format: MP3/WMA -> FLAC
* Sample Rate: 44,100 Hz -> 16,000 Hz
* Channels: 2 (stereo) -> 1 (mono)

First, open [preprocessing/01_generate_flac.py](preprocessing/01_generate_flac.py) and edit the global variables: `INPUT_DIR`, `OUTPUT_DIR`, and `ffmpeg`. Then, run:

```bash
python preprocessing/01_generate_flac.py
```

If running on NERO or other compute cluster, submit your job with Slurm (see [preprocessing/01_slurm.sh](preprocessing/01_slurm.sh)).:

```bash
sbatch scripts/01_slurm.sh
squeue
```

The output flac files will be placed in `OUTPUT_DIR`.

### 2.3   Ground Truth JSON

The ground truth files are currently in TXT files, as delivered by the annotators. We need to convert these to JSON as a standardization step.

1. Ensure that `gt_dir` is correct inside `preproc/config.py`.
2. Run the following inside `preproc/`.

```bash
python 03_create_gt_json.py OUTPUT_DIR
```

where `OUTPUT_DIR` is the target location to place the new, ground truth JSON files.

## 3   Speech-to-Text with Google Cloud

### 3.1   Prerequisites
First, enter your GCloud key and bucket information in [gcloud/config.py](gcloud/config.py).

- **Google Cloud API Key**. This should be a *service account*. The key should have permissions to Google Cloud storage and Google Speech-to-Text API. Easiest but not-security-recommended solution would grant the service account with *Project Owner* status/permissions.
- **Bucket Name**. This bucket should already exist. Our script will not create a bucket. The files will be uploaded to this bucket.

Second, install the python dependencies.
```bash
pip install -r requirements.txt
```

### 3.2   Upload to Google Cloud

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

### 3.3   Speech-to-Text

Now that we have a single bucket containing only flac files, we can run transcription and diarization.

```bash
cd gcloud
python 02_transcribe.py OUTPUT_DIR
```

where `OUTPUT_DIR` is your desired *local* folder where to store the json transcription results. This script will also print the transcription progress.

## 4   Evaluation

The final step is to compute metrics between the Google Speech API and the ground truth. At this point, we should have two directories (and what we named them):

1. `machine`: Contains the Google Speech API output transcriptions as JSON format.
2. `gt`: Contains the ground truth transcriptions as JSON format (see Section 2.3).

These JSON files serve as the final output of our algorithm. We will never edit them again. All downstream metrics will be computed by reading these JSON files.

### 4.1   Phrase vs Session Level

Each ground truth transcript contains multiple *phrases*. Each phrase is a short set of words, spoken either by the therapist or the patient (the ground truth provides the speaker identity, i.e., patient/therapist for each phrase.

A *session* is composed of multiple phrases from a single speaker. For example, "session-level patient" refers to the concatenation of all phrases said by the patient in a given therapy session. This can also be done for the therapist.

Before computing any metrics, we must split the ground truth and prediction files into phrases.

```bash
export PYTHONPATH=.
python evaluation/phrase_level.py MACHINE_DIR GT_DIR
```

where `MACHINE_DIR` is the directory of ASR-transcribed JSON files and `GT_DIR` is the directory of GT JSON files.

Running the above script will produce `results/session.csv`, `results/phrase.csv`, and `results/text.txt`. The `text.txt` file contains each phrase, split by speaker as well. The `session.csv` and `phrase.csv` files contain the quant metrics described below.

### 4.2   WER, BLEU, GLEU

We use an initial set of metrics for computing ASR performance.

1. Word Error Rate (WER)
2. Bilingual Evaluation Understudy ([BLEU](https://en.wikipedia.org/wiki/BLEU))
3. Google's Evaluation Understudy ([GLEU](https://www.nltk.org/_modules/nltk/translate/gleu_score.html))

### 4.3   Clinical Unigrams

Four clinicians hand-picked words that were critical for detecting mental health issues. We refer to these words as "clinical unigrams". The idea is that WER, BLEU, give equal weight to words, when in reality, words such as "suicide" are much more important and should be reflected in the performance metrics.

```bash
python evaluation/clinical_unigrams.py TEXT_FILE
```

where `TEXT_FILE` is the output from `evaluation/phrase_level.py`. It consists of the ground truth and predicted transcription, one on different lines. It is visually easy to compare sentences, as well as loading for automated processing.

One naive approach is to compute the frequency of such unigrams in the ground truth, both at the phrase and session level. Then we compare the frequency in the predicted transcription.

### 4.4   Embeddings

Extracting embeddings is time consuming. Therefore, we first extract embeddings, save them to disk, then compute similarity metrics on these saved embeddings.

#### 4.4.1   Word2Vec

Download Word2Vec: [[Google Drive](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)] [[website](https://code.google.com/archive/p/word2vec/)] (1.5 GB)

Once downloaded, uncompress the file: `gunzip GoogleNews-vectors-negative300.bin.gz`

Take note of the location of the .bin file and update the `WORD2VEC_MODEL_FQN` variable inside `evaluation/embeddings/config.py`.

#### 4.4.2   GloVe

Download GloVe: [[zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)] [[website](https://nlp.stanford.edu/projects/glove/)] (Common Crawl, 840B tokens, 2.0 GB)

Once downloaded, uncompress it.

#### 4.4.3   BERT

Extracting BERT embeddings involves two parts: The server and the client.

1. Install the bert-as-a-service python package **and** download a BERT model: https://github.com/hanxiao/bert-as-service
2. Start the BERT embedding server: `evaluation/embeddings/server/start.sh`
3. Extract embeddings for each sentence: `evaluation/embeddings/02_bert_encode.py`