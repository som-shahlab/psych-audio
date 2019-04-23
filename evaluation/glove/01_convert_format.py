"""
After downloading the GloVe files from the Stanford website, run this script
to convert it to Gensim's word2vec format. The word2vec *format* can also support Glove.

Download GloVe:
https://nlp.stanford.edu/projects/glove/
"""
import os
import sys
import argparse
import gensim.test.utils
import gensim.models
from gensim.scripts.glove2word2vec import glove2word2vec


def main(args):
	if not os.path.exists(args.input_fqn):
		print(f'Does not exist: {args.input_fqn}')
		sys.exit(1)

	glove_file = gensim.test.utils.datapath(args.input_fqn)
	tmp_file = gensim.test.utils.get_tmpfile(args.output_fqn)
	_ = glove2word2vec(glove_file, tmp_file)
	model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
	print(model)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input_fqn', type=str, help='Glove model as a txt file.')
	parser.add_argument('output_fqn', type=str, help='Where to store the converted Glove model.')
	main(parser.parse_args())
