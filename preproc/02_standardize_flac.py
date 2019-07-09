"""
This script takes existing flac files and standardizes them according
to our metadata tsv file. This includes joining multiple flac files into
a single audio file (e.g., a single therapy session was split into two audio files).

Make sure to have finished running `preproc/01_generate_flac.py` before running this script.
"""
import os
import shutil
import string
import argparse
import subprocess
import pandas as pd
import preproc.util
import preproc.config


def main(args):
    # Maintains the original filename -> hashed filename mapping.
    meta = pd.read_csv(preproc.config.meta_fqn, sep="\t")

    # Used for concating two files.
    cmd_template = string.Template(
        'ffmpeg -i "$infile1" -i "$infile2" -c:a flac -ac 1 -ar 16000 '
        "-filter_complex '[0:0][1:0]concat=n=2:v=0:a=1[out]' -map '[out]' $outfile"
    )

    # For each metadata row, copy and rename the audio file.
    for _, row in meta.iterrows():
        hash = row["hash"]
        path = str(row["audio_path"])

        if path == "nan":
            continue
        # Handle the case where we need to concat two files.
        elif ";" in path:
            out_fqn = os.path.join(args.output_dir, f"{hash}.flac")
            if os.path.exists(out_fqn):
                continue

            # We have 2 paths. Need to concat.
            paths = path.split(";")
            input_files = []
            for _, path in enumerate(paths):
                filename = preproc.util.remove_extension(os.path.basename(path))
                fqn = os.path.join(args.input_dir, f"{filename}.flac")
                input_files.append(fqn)

            assert len(input_files) == 2
            cmd = cmd_template.substitute(
                infile1=input_files[0], infile2=input_files[1], outfile=out_fqn
            )
            subprocess.run(cmd, shell=True)
            print(f"(Concat) {out_fqn}")
        else:
            # Handle the case where there's only one file. Simply copy it to our target directory.
            filename = preproc.util.remove_extension(os.path.basename(path))
            if filename in preproc.config.malformed_files:
                print(f"Skipping malformed: {filename}")
                continue
            source_fqn = os.path.join(args.input_dir, f"{filename}.flac")
            dest_fqn = os.path.join(args.output_dir, f"{hash}.flac")
            if os.path.exists(dest_fqn):
                continue
            shutil.copy(source_fqn, dest_fqn)
            print(source_fqn, dest_fqn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir", type=str, help="Directory which contains all flac files."
    )
    parser.add_argument(
        "output_dir", type=str, help="Location to place the new flac files."
    )
    args = parser.parse_args()
    main(args)
