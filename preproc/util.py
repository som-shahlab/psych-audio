"""
This file contains helper functions for the other pre-processing files.
"""
import os
import re
import string

# Used for converting numbers to words.
num2words1 = {
	0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
	6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
	11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
	15: 'fifteen', 16: 'sixteen', 17: 'seventeen',
	18: 'eighteen', 19: 'nineteen'}
num2words2 = [
	'twenty', 'thirty', 'forty', 'fifty', 'sixty',
	'seventy', 'eighty', 'ninety']
ordinals = {
	'1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth',
	'5th': 'fifth', '6th': 'sixth', '7th': 'seventh', '8th': 'eighth',
	'9th': 'ninth', '10th': 'tenth'}

# Remove these words, either because Speech API does not transcribe them,
# or they are considered stop words.
remove_words = [
	'um', 'mm', 'mmhmm', 'mmmm', 'uh', 'uhhuh', 'hmm', 'umhmm', 'huh', 'mmmhmm', 'aa', 'gonna', 'umm', 'hmhmm', 'uhuh',
	'mmm', 'ah', 'uhhmm', 'ii', 'uhhum', 'mmhm', 'hm', 'ccaps', 'gotta', 'imim', 'eh', 'ugh', 'gotcha', 'hmmm', 'un']
remove_symbols = ['—', '’', '“']


def canonicalize_word(word: str) -> str:
	"""
	Canonicalizes a single word.
	- Converts numbers 0-99 into words.
	- Converts to lower-case.
	- Removes scrubbed data (denoted by brackets, e.g., [laugh])
	- Removes punctuation.
	Note: Canonicalized, scrubbed words will become the empty string.

	From: https://stackoverflow.com/questions/19504350/how-to-convert-numbers-to-words-in-python
	"""
	word = word.lower()

	# Remove scrubbed data. Needs to happen before punctuation removal.
	if '[' in word and ']' in word:
		return ''
		# word = re.sub('\[.*\]|\s-\s.*', '', word)

	# Remove punctuation and other symbols.
	word = word.translate(str.maketrans('', '', string.punctuation))
	for sym in remove_symbols:
		word = word.replace(sym, '')

	# Convert numbers to words.
	if word.isdigit():
		word = digits_to_words(word)
	if word in ordinals:
		word = ordinals[word]

	# Remove everything except letters.
	# clean_word = ''
	# for c in word:
	# 	if c.isalpha():
	# 		clean_word += c

	# Remove stop words.
	if word in remove_words:
		return ''

	return word


def canonicalize_sentence(text: str) -> str:
	"""
	Canonicalizes a full sentence.
	:param text: Input text as a string.
	:return: Cleaned-up text.
	"""
	# Convert to lower-case.
	text = text.lower()

	# Merge multiple spaces.
	text = re.sub(' +', ' ', text).strip()
	text = text.strip()

	# Process each word.
	tokens = text.split(' ')
	for i in range(len(tokens)):
		tokens[i] = canonicalize_word(tokens[i])
	text = ' '.join(tokens)

	text = text.lower()

	return text


def digits_to_words(digits: str) -> str:
	"""
	Converts a number, represented as digits, into English words.
	e.g. 12 -> twelve, 48 -> forty eight
	:param digits: String containing the number.
	:return: String of the words.
	"""
	integer = int(digits)
	if 0 <= integer < 19:
		return num2words1[integer]
	elif 20 <= integer <= 99:
		tens, below_ten = divmod(integer, 10)
		return num2words2[tens - 2] + ' ' + num2words1[below_ten]
	else:
		return ''


def metadata_file_is_clean(fqn: str) -> bool:
	"""
	Checks whether teh metadata file is valid.
	- All rows are tab-delimited.
	- All rows contain valid audio files. Note that some rows have two associated files.
	- All rows contain a valid SHA256 hash.
	- All rows have 36 columns.

	:param fqn: Full path to the metadata file.
	:return: True if metadata file is correct. False otherwise.
	"""
	# Check if it exists.
	if not os.path.exists(fqn):
		print(f'File does not exist: {fqn}')
		return False

	# Open and check each line.
	with open(fqn, 'r') as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			# Skip the first line since it contains headers.
			if i == 0:
				continue

			# Check if line is valid.
			tokens = line.strip().split('\t')
			if len(tokens) != 36:
				print(f'Line {i+1} does not have 36 columns.')
				return False

			fqns = tokens[32].split(';')
			# It is possible for this row to have zero filenames.
			if len(fqns) == 1 and fqns[0] == '':
				continue
			# If this row has non-empty audio filenames.
			for fqn in fqns:
				if not os.path.exists(fqn):
					print(f'Line {i+1} audio does not exist: {fqn}')
					return False

	return True


def remove_extension(filename: str) -> str:
	"""
	Removes the mp3 or wma extension from a string.

	:param filename: Filename to parse.
	:return: Filename but without the extension.
	"""
	for ext in ['.wma', '.mp3', '.wav']:
		filename = filename.replace(ext, '')
		filename = filename.replace(ext.upper(), '')
	return filename
