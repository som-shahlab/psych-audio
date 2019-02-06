#!/bin/bash
#
#SBATCH --job-name=gen-flac
#
#SBATCH --time=60:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G

srun /home/akhaque/anaconda/bin/python /home/akhaque/psych-audio/scripts/01_generate_flac.py --n_threads 4