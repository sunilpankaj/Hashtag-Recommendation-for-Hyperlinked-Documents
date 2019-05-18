"""
	Program: main code to recommend hashtags given a post (end-to-end code)
"""
import os
print "0"
import csv
print "1"
import LTModel_test
print "2"
print "3"
import similarDesc_test
print "4"
import similarContent_test
print "5"
import d2v_test
print "6"
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
#import sim_desc
print "7"
#import sim_content
print "8"
#import RWR
print "9"
#import LTModel
print "10"
#import d2v
#from d2v import doc2vect
import time
#Parameter
K = 10

print "IMPORT LIBRATIES DONE " 
def FileSave(filename,content):
	with open(filename, "a") as myfile:
		myfile.write(content)

def similarDesc_t(row):
	res = similarDesc_test.similarDesc1(row,1)
	return res

def similarContent_t(row):
	res = similarContent_test.similarContent1(row,1)
	return res

def LTRes_t(row):
	#entities = [r.strip() for r in row[8].split(";")]
	res = LTModel_test.LTRes1("/home/tdbms06/Code_tweet/training_tweet.csv",row,5)
	return res

def simd2v_t(row):
	res = d2v_test.simd2v("/home/tdbms06/Code_tweet/training_tweet.csv",row,2)
	return res




def feature_cre(row):
	start_time = time.time()
	list_descRes = similarDesc_t(row)
	print type(list_descRes)
	#print list_descRes
	print " sim desc complete"

	list_contentRes = similarContent_t(row)
	#print list_contentRes
	print " sim content complete"
	#list_domainRes = 
	#list_rwrRes = RandomWalk("testing.csv")
	list_ltRes= list(LTRes_t(row))

	print " language transaltion complete"
	list_d2v = simd2v_t(row)
	#print list_d2v
	print " doc 2 vec complete"

	act_tweet = []
	kact = row[7];
	kact = ''.join(kact).replace(' ',"").split(';')
	kact = kact[0].split(',')
	for i in kact:
		act_tweet.append(i.lstrip())

	print " getting the list of each method done "
	print("--- %s hashtag prediction of 3 method in  ---" % (time.time() - start_time))

	pos_hash = []
	neg_hash = []

	count = 0
	p11 = set(list_contentRes)
	p12 = set(list_descRes)
	p13 = set(list_ltRes)
	p14 = set(list_d2v)
	h1 = set(act_tweet)

	# Intersection to find pos hag
	i1 = p11.intersection(h1)
	i2 = p12.intersection(h1)
	i3 = p13.intersection(h1)
	i4 = p14.intersection(h1)

	temp = i1.union(i2,i3,i4)
	#temp = i1.union(i2,i3)
	#print "temp type ::",type(temp)
	#print type(list(temp))
	pos_hash = list(temp)

	# difference to find neg hashtags
	n1 = p11.difference(i1)
	n2 = p12.difference(i2)
	n3 = p13.difference(i3)
	n4 = p14.difference(i4)

	temp1 = n1.union(n2,n3,n4)
	#temp1 = n1.union(n2,n3)
	neg_hash = list(temp1)
	#print pos_hash
	#print neg_hash
	#print len(pos_hash)
	count += (len(pos_hash) + len(neg_hash))
	#print count
	X_t = np.zeros((count, 4))
	Y_t = np.zeros((count, 2))

	index = 0
	pred_hash = []

	for j in pos_hash:
		pred_hash.append(j)
		if(j in list_contentRes):
			X_t[index][0] = 1
		if (j in list_descRes):
			X_t[index][1] = 1
		if (j in list_ltRes):
			X_t[index][2] = 1
		
		if(j in list_d2v):
			X_t[index][3] = 1
		Y_t[index][0] = 1
		index += 1

	for j in neg_hash:
		pred_hash.append(j)
		if(j in list_contentRes):
			X_t[index][0] = 1
		if (j in list_descRes):
			X_t[index][1] = 1
		if (j in list_ltRes):
			X_t[index][2] = 1
		
		if(j in list_d2v):
			X_t[index][3] = 1 
		Y_t[index][0] = -1
		index += 1
	
	return X_t,Y_t,pred_hash



#x,y = feature_cre(fil)

'''

if(__name__ == "__main__"):
	with open("testing_eco_100.csv", "r") as file:
		reader = csv.reader(file)
		rows = list(reader)
		# total number of posts = 501
		#test_index = range(450,460)
		test_index = range(1,4)
		for j in test_index:
			#print "j :: ",j
			row = rows[j]
			print row
			feature_cre(row)
			#list of actual hashtags
			kact = row[7];
			kact = ''.join(kact).replace(' ',"").split(';')
			print type(kact)
			test_tweet = [];

'''
   
