"""
	Program: main code to recommend hashtags given a post (end-to-end code)
"""
import os
print "0"
import csv
import networkx as nx
import operator
from collections import Counter
import numpy as np
import itertools
from sklearn import svm, linear_model, cross_validation
from collections import Counter
from sklearn import metrics
import spacy
from sklearn.metrics import confusion_matrix
nlp = spacy.load("en")
import sim_desc
import sim_content
import LTModel
import d2v
import time
#Parameter
K = 10

print "IMPORT LIBRATIES DONE " 
def FileSave(filename,content):
	with open(filename, "a") as myfile:
		myfile.write(content)

def similarDesc1(fil):
	#res = similarDesc.simDesc("training.csv","testing.csv")
	tot_predhas,actualList = sim_desc.similarDesc("/home/tdbms06/Code_tweet/training_tweet.csv",fil,1)
	return tot_predhas,actualList

def similarContent1(fil):
	tot_predhas,actualList = sim_content.similarCon("/home/tdbms06/Code_tweet/training_tweet.csv",fil,1)
	return tot_predhas,actualList

'''def domainLink1(row):
	res = domainLink.domain("politics_refined.csv",K)
	return res

def RandomWalk(fil):
	#entities = [r.strip().split('#$#')[0] for r in row[8].split(";")]
	res = RWR.rwr("training_eco.csv",fil,5)
	return res
'''
def LTRes1(fil):
	#entities = [r.strip() for r in row[8].split(";")]
	tot_predhas,actualList = LTModel.LT("/home/tdbms06/Code_tweet/training_tweet.csv",fil,5)
	return tot_predhas,actualList

def simd2v(fil):
	#dis = row[5]
   print "sim enter"
   tot_predhas,actualList = d2v.doc2vect("/home/tdbms06/Code_tweet/training_tweet.csv",fil,2)
   return tot_predhas,actualList

def feature_cre(fil):
	start_time = time.time()
	list_descRes,act_descRes = similarDesc1(fil)
	print " sim desc complete"

	list_contentRes,act_contentRes = similarContent1(fil)
	print " sim content complete"
	#list_domainRes = 
	#list_rwrRes = RandomWalk("testing.csv")
	list_ltRes,act_ltRes = LTRes1(fil)
	print " language transaltion complete"
	list_d2v,act_d2v = simd2v(fil)
	print " doc 2 vec complete"

	print " getting the list of each method done "
	print("--- %s hashtag prediction of 3 method in  ---" % (time.time() - start_time))


	pos_hash = []
	neg_hash = []

	count = 0
	for i in range(len(list_ltRes)):
		p11 = set(list_contentRes[i])
		p12 = set(list_descRes[i])
		p13 = set(list_ltRes[i])
		p14 = set(list_d2v[i])
		h1 = set(act_contentRes[i])

		# Intersection to find pos hag
		i1 = p11.intersection(h1)
		i2 = p12.intersection(h1)
		i3 = p13.intersection(h1)
		i4 = p14.intersection(h1)

		temp = i1.union(i2,i3,i4)
		#temp = i1.union(i2,i3)
		pos_hash.append(list(temp))

		# difference to find neg hashtags
		n1 = p11.difference(i1)
		n2 = p12.difference(i2)
		n3 = p13.difference(i3)
		n4 = p14.difference(i4)

		temp1 = n1.union(n2,n3,n4)
		#temp1 = n1.union(n2,n3)
		neg_hash.append(list(temp1))

		count += (len(pos_hash[i]) + len(neg_hash[i]))


	X = np.zeros((count, 4))
	Y = np.zeros((count, 2))

	index = 0

	for k in range(len(list_ltRes)):
		for j in pos_hash[k]:
			if(j in list_contentRes[k]):
				X[index][0] = 1
			if (j in list_descRes[k]):
				X[index][1] = 1
			if (j in list_ltRes[k]):
				X[index][2] = 1
			
			if(j in list_d2v[k]):
				X[index][3] = 1

			Y[index][0] = 1
			Y[index][1] = k
			index += 1

		for j in neg_hash[k]:
			if(j in list_contentRes[k]):
				X[index][0] = 1
			if (j in list_descRes[k]):
				X[index][1] = 1
			if (j in list_ltRes[k]):
				X[index][2] = 1
			
			if(j in list_d2v[k]):
				X[index][3] = 1 
			Y[index][0] = -1
			Y[index][1] = k
			index += 1

	return X,Y

#x,y = feature_cre(fil)



