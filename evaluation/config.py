from typing import Dict

# Path to Google's pre-trained word2vec model (.bin file)
WORD2VEC_MODEL_FQN: str = "models/word2vec/GoogleNews-vectors-negative300.bin"

# Path to Stanford's pre-trained GloVe model (.txt file)
GLOVE_MODEL_FQN: str = "models/glove/glove.840B.300d.txt"

# Dimension of each embedding.
F: Dict[str, int] = {"word2vec": 300, "glove": 300, "bert": 1024}

# Path to the metadata file.
META_FQN = "results/_phq9_diffs_with_paths.tsv"

# Path to the paired.json corpus and predictions file.
PAIRED_FQN = "results/paired7.json"

# Path to the paired.json corpus and predictions file.
TABLE2_FQN = "results/table2.tsv"

# Path to the table 3 intermediate file.
TABLE3_FQN = "results/table3.tsv"

# Path to the PHQ term file.
PHQ_TERM_FQN = "data/clinical-terms-v3.tsv"

# Path to the vocabulary file of all possible words.
VOCAB_FQN = "data/english.txt"

# Which dimensions to analyze. That is, each item in the below list
# will be a subsection of the table. i.e., Gender will be broken down into male/female (its unique values).
DIMENSIONS = ["gender_imputed", "Num_sess", "Age_ses1", "PHQ9_total_ses"]
