#!/bin/bash
#
#SBATCH --job-name=gen-flac
#
#SBATCH --time=60:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G

for ARGUMENT in "$@"
do
    
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    
    case "$KEY" in
            PYTHON_DIR)      PYTHON_DIR=${VALUE} ;;
            FLAC_SCRIPT)    FLAC_SCRIPT=${VALUE} ;;     
            OUTPUT_DIR)     OUTPUT_DIR=${VALUE} ;;
            *)
    esac

done

echo "PYTHON_DIR = $PYTHON_DIR"
echo "FLAC_SCRIPT = $FLAC_SCRIPT"
echo "OUTPUT_DIR = $OUTPUT_DIR"

srun $PYTHON_DIR $FLAC_SCRIPT $OUTPUT_DIR --n_threads 4