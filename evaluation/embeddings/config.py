from typing import Dict

# Path to Google's pre-trained word2vec model (.bin file)
WORD2VEC_MODEL_FQN: str = '/vol0/psych_audio/ahaque/models/word2vec/GoogleNews-vectors-negative300.bin'

# Path to Stanford's pre-trained GloVe model (.txt file)
GLOVE_MODEL_FQN: str = '/vol0/psych_audio/ahaque/models/glove/glove.840B.300d.txt'

# Dimension of each embedding.
F: Dict[str, int] = {'word2vec': 300, 'glove': 300, 'bert': 1024}

# Max BERT sequence length (words), as specified in `server/start.sh`.
SEQ_LEN = 100

# Location of the saved embeddings (npz files).
NPZ_DIR = '/vol0/psych_audio/ahaque/psych-audio/results/embeddings'

# Where to save the output csv distance files.
DISTANCES_DIR = '/vol0/psych_audio/ahaque/psych-audio/results/dists'

# Path to the metadata file.
META_FQN = '/vol0/psych_audio/jasa_format/metadata.tsv'

# Which dimensions to analyze. That is, each item in the below list
# will be a subsection of the table. i.e., Gender will be broken down into male/female (its unique values).
DIMENSIONS = [
	'Gender_ses1',
	'Num_sess',
	'Age_ses1',
	'PHQ9_total_ses',
]
