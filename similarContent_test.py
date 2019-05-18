from __future__ import division
import os
import csv
import re
import string
from nltk.corpus import stopwords
from nltk import word_tokenize

import string
import math
# Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

content = []
desc = []
tag = []
entity =[]
tags_list = []

precision = 0.0
recall = 0.0
count = 0

def simContent(fil,test,hashtags,k):
	global precision,recall,count
	'''
	if("" in hashtags):
		hashtags.remove("")
	if "'#'" in hashtags:
		hashtags.remove("'#'") '''
	content_train = []
	tags_list_train = []
	with open("training_eco.csv", "r") as file:
		reader = csv.reader(file)
		next(reader, None)
		for row in reader:
			name = []
			_type = []
			content_train.append(row[6])
			#tag = [r.strip()[1:-1] for r in row[7].split(";")]
			tag = [r.strip() for r in row[7].split(";")]
			'''
			x = ""
			if("" in tag):
				tag.remove("")
			for i in range(len(tag)):
				for j in range(len(name)):
					if tag[i] == name[j] and _type[j] == "Person":
						#print tag[i]
						break
				x = x + tag[i] + ";" 
			tags_list_train.append(x)
			'''
			tag = tag[0].split(',')
			tag_lstrip = []
			for i in tag:
				tag_lstrip.append(i.lstrip())
			tags_list_train.append(tag_lstrip)

	#print type(con)
	content_test = []
	content_test.append(test)
	tags_list_test = hashtags
	
	text_train = content_train
	text_test = content_test

	# create the transform
	vectorizer = TfidfVectorizer()

	# tokenize and build vocab
	data = vectorizer.fit_transform(text_train)

	#Test vector
	vector = vectorizer.transform(text_test)

	tfidf_comparisons = []
	for i in range(len(text_test)):
		sims = cosine_similarity(vector[i], data)
		tfidf_comparisons.append([sims,i])

	#tagsList = set()
	tagsList = []
	for p in range(len(text_test)):
		p_tfidf = tfidf_comparisons[p][0][0]
		srt = p_tfidf.argsort()[-k:][::-1]
		#print "srt printed ::",srt
		for i in range(k):
			'''
			hash_list = []
			t = tags_list_train[srt[i]].split(";")
			s = t[0]
			hash_list = s.split(',')
			'''
			tagsList.extend(tags_list_train[srt[i]])
			
		if '' in tagsList:
			tagsList.remove('')
		if "'#'" in tagsList:
			tagsList.remove("'#'")
	tagsList = set(tagsList)    
	temp=hashtags[0].split(',')
	emp = []
	A = []
	for i in temp:
		A.append(i.lstrip())
	if (len(A) == 0):
		return emp
	else:
		'''
		A = set(A)
		#print A
		#print tagsList
		#print A.intersection(tagsList)
		if len(tagsList) != 0:
			#print "precision",len(A.intersection(tagsList))/(len(tagsList)+0.0)
			precision = precision + len(A.intersection(tagsList))/(len(tagsList)+0.0)
		else:
			precision = precision + 0
		#print "recall",len(A.intersection(tagsList))/(len(A)+0.0)
		recall = recall + len(A.intersection(tagsList))/(len(A)+0.0)
		count = count + 1
		print "Total precision is " , precision/count
		print "Total recall is ", recall/count '''
		return list(tagsList)
	

def similarContent1(row,k):
	hashtags = [r.strip() for r in row[7].split(";")]
	#print hashtags
	res = simContent("training_eco.csv",row[6],hashtags,k)
	return res
'''
if(__name__ == "__main__"):
	with open("testing_eco.csv", "r") as file:
		reader = csv.reader(file)
		next(reader, None)
		count = 0
		for row in reader:
			#descRes = similarDesc1(row)
			contentRes = similarContent1(row)
			count += 1
			print count
			#domainRes = domainLink1(row)
			#rwrRes = RandomWalk(row)
			#ltRes = LTRes1(row)
			#break
		print "Total precision is " , precision/count
		print "Total recall is ", recall/count  
'''

