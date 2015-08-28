#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :
from __future__ import print_function
import sys
import gc
import resource
import re
import logging
import time
import itertools
from datetime import timedelta

from optparse import OptionParser

import numpy as np

from scipy import sparse as sp

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lexicon import Lexicon

import nltk
from nltk.corpus import mac_morpho

parser = OptionParser(usage="%prog [options] <datasetdir>")
parser.add_option("-e", "--encoding", dest="encoding", default="latin_1", help="Dataset encoding")
parser.add_option("-r", "--initial-ranking", dest="ranking_method", default="cosine_similarity", help="Initial ranking method (cosine_similarity, accuracy) Default: cosine_similarity")

if sys.stdout.encoding == None:
	print("Fixing stdout encoding...")
	import codecs
	import locale
	# Wrap sys.stdout into a StreamWriter to allow writing unicode.
	sys.stdout = codecs.getwriter(locale.getpreferredencoding())(sys.stdout)

(options, args) = parser.parse_args()

#if len(args) == 0:
	#parser.print_help()
	#sys.exit()



class ElapsedFormatter():
	
	def __init__(self):
		self.start_time = time.time()
	
	def format(self, record):
		elapsed_seconds = record.created - self.start_time
		#using timedelta here for convenient default formatting
		elapsed = timedelta(seconds = elapsed_seconds)
		return "[%s][RAM: %.2f MB] %s" % (str(elapsed)[:-3], (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), record.getMessage())

#add custom formatter to root logger for simple demonstration
handler = logging.StreamHandler()
handler.setFormatter(ElapsedFormatter())
logging.getLogger().addHandler(handler)

log = logging.getLogger('main')
log.setLevel(logging.DEBUG)

with open("input.txt") as f:
	text = f.readlines()
	inputs = [i.replace("\n", "").lower().split(";") for i in text]
	log.info(inputs)

#dataset_folder = args[0]

lexicon = Lexicon("Portuguese (Brazil)/Dela/")

def get_candidates(sentences):
	candidates_simple = set()
	candidates_med = set()
	candidates_full = set()
	tokenizer = re.compile('\w+')
	for s in sentences:
		sent_words = tokenizer.findall(s)
		pos_full = []
		pos_med = []
		pos_simple = []
		for w in sent_words:
			lemmas = lexicon.get_lemmas(w)
			pos_full += [set([p[1] for p in lemmas])]
			pos_med += [set([p[1].split(":")[0] for p in lemmas])]
			pos_simple += [set([p[1].split(":")[0].split("+")[0] for p in lemmas])]
			#print(w, lemmas)
			#print(pos_med)
			#print(pos_simple)
		
		if len(candidates_simple) == 0:
			#print("TESTE",pos_simple)
			candidates_simple = set(itertools.product(*pos_simple))
			candidates_med = set(itertools.product(*pos_med))
			candidates_full = set(itertools.product(*pos_full))
		else:
			candidates_simple = candidates_simple.intersection(set(itertools.product(*pos_simple)))
			candidates_med = candidates_med.intersection(set(itertools.product(*pos_med)))
			candidates_full = candidates_full.intersection(set(itertools.product(*pos_full)))
		#print("ITERTOOLS")
		#print(candidates_simple)
	return candidates_simple, candidates_med, candidates_full

sentences = [s[1] for s in inputs]

log.info("Loading Mac-Morpho Tagged Sents...")
tsents = list(mac_morpho.tagged_sents())


def simplify_tag(t):
	if "+" in t:
		t = t[t.index("+")+1:]
	
	if t == "ART":
		return "DET"
	
	return t

log.info("Simplifyng POS Tags...")
tsents = [[(w.lower(),simplify_tag(t)) for (w,t) in sent] for sent in tsents if sent]

train = tsents
test = tsents[:300]
log.info("Training POS Taggers...")
tagger0 = nltk.DefaultTagger('N')
tagger1 = nltk.UnigramTagger(train, backoff=tagger0)
tagger2 = nltk.BigramTagger(train, backoff=tagger1)

#log.info("Evaluate tagger")
#print(tagger2.evaluate(test))

#log.info("TAGSET")
#tags = [simplify_tag(tag) for (word,tag) in mac_morpho.tagged_words()]
#fd = nltk.FreqDist(tags)
#print(fd.keys())

tokenizer = re.compile('\w+')
for input_id, s in enumerate(sentences):
	log.info("Sentence: %s" % (s))
	candidates_simple, candidates_med, candidates_full = get_candidates([s]) 
	#print(candidates_simple)
	tagged_sent = [tagger2.tag([w])[0][1] for w in tokenizer.findall(s)]
	#print(s, tagged_sent)
	candidates_simple = np.array(list(candidates_simple))
	tagged_sent = np.array(tagged_sent)
	gram_acc = candidates_simple == tagged_sent
	#print(gram_acc)
	gram_acc = gram_acc.astype(np.float64).sum(axis=1) / gram_acc.shape[1]
	#print(gram_acc)
	
	log.info("Vectorizing...")
	count_vect = CountVectorizer(dtype=np.float64, token_pattern='\w+')
	X = [" ".join(tokens) for tokens in candidates_simple]
	#print(X)
	X_vect = count_vect.fit_transform(X)
	#print(X_vect.todense())
	tagged_sent_vect = count_vect.transform([" ".join(tagged_sent)])[0]
	#print(tagged_sent_vect)
	#print(X[0])
	#print(" ".join(tagged_sent))
	#print(tagged_sent_vect.todense())
	log.info("(%d, %d)" % (X_vect.shape[0],X_vect.shape[1]))
	
	gram_sim = cosine_similarity(X_vect, tagged_sent_vect)
	#print(gram_sim)
	
	if options.ranking_method == "cosine_similarity":
		log.info("Using cosine_similarity ranking...")
		gram_rank = gram_sim
	elif options.ranking_method == "accuracy":
		log.info("Using accuracy ranking...")
		gram_rank = gram_acc
	else:
		log.warning("Unknown ranking method %s ignored, using cosine_similarity")
		gram_rank = gram_sim
	
	top_idx = np.argmax(gram_rank)
	top_gram = candidates_simple[top_idx]
	#print(top_gram)
	
	log.info("%s: Writing results..." % (inputs[input_id]))
	with open("gram-%d.txt" % (input_id), 'w') as f:
		f.write("%s\n" % (inputs[input_id]))
		f.write("Meta: %s\n" % (" ".join(tagged_sent)))
		sorted_grams_idx = np.argsort(-gram_rank, axis=0)
		#print("sorted_grams_idx",sorted_grams_idx)
		for i, gram_idx in enumerate(sorted_grams_idx):
			gram = candidates_simple[gram_idx][0]
			#print(gram_idx, gram)
			f.write("%d: %s - %.03f\n" % (i, " ".join(gram), gram_rank[gram_idx]))
log.info("Finished")
	