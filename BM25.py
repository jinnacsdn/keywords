#!usr/bin/python 

import sys
import math

def get_doc_count(lines):
	"""Get doc count.

	"""
	return len(lines)

def get_avg_doc_len(lines):
	"""Get all docs average words.

	"""
	sum = 0;
	for line in lines:
		sum += len(line.split())
	return 1.0*sum/len(lines)

def get_idfs(lines, doc_count):
	"""Get words idf.

	"""
	word_docs = {}
	for line in lines:
		words = set(line.strip().split())
		for word in words:
			if word in word_docs:
				word_docs[word] += 1.0
			else:
				word_docs[word] = 1.0
	
	idf = {}
	for line in lines:
		words = set(line.strip().split())
		for word in words:
			if word not in word_docs:
				idf[word] = 0.0
			else:
				idf[word] = math.log((doc_count + 0.5)/(word_docs[word]+0.5))
	return idf

def get_idf(idf, word):
	"""Get word idf.

	"""
	if word not in idf:
		return 0.0
	else:
		return idf[word]

def get_tf(word, sentence, avgdl, k1=2.0, b=0.75):
	"""Get the tf of word.

	"""
	words = sentence.strip().split()
	dl = len(words)
	f = words.count(word)
	tf = f * (k1 + 1) / (f + k1*(1-b + b * dl/avgdl))
	return tf

def get_key_words(sentence, idf, avgdl, top=1, k1=2.0, b=0.75):
	"""
	"""
	words = set(sentence.strip().split())
	word_score_list = []
	for word in words:
		word_score_list.append((word, get_tf(word, sentence, avgdl, k1, b)*get_idf(idf, word)))
	word_score_list.sort(key = lambda x:x[1], reverse=True)
	return word_score_list[0:top]
				
if __name__ == '__main__':
	"""Main function.

	"""
	f = open(sys.argv[1], 'r')
	lines = f.readlines()
	f.close()

	n = get_doc_count(lines)
	avgdl = get_avg_doc_len(lines)
	idf = get_idfs(lines, n)

	for line in lines:
		keywords = get_key_words(line.strip(), idf, avgdl, top=2, k1=2.0, b=0.75)
		print line.strip(), keywords
