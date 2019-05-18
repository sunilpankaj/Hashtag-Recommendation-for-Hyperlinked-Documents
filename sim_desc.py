from __future__ import division
import os
import csv
import re
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
import time
import string
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



def simDesc(training_file,testing_file,k):
    # performing function
    #df = pd.read_csv("content_20K.csv", header=None)
    
    #text = df[0].tolist()
    
    #text_test = df_test[0].tolist()
    #text_train = text[:-(len(text)-x)]
    #text_test = text[x:]

    desc_train = []
    tags_list_train = []
    with open(training_file, "r") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            name = []
            _type = []
            desc_train.append(row[5])
            #tag = [r.strip()[1:-1] for r in row[7].split(";")]
            tag = [r.strip() for r in row[7].split(";")]
            tag = tag[0].split(',')
            tag_lstrip = []
            for i in tag:
                tag_lstrip.append(i.lstrip())
            tags_list_train.append(tag_lstrip)
           
    
    desc_test = []
    tags_list_test = []
    with open(testing_file, "r") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            name = []
            _type = []
            desc_test.append(row[5])
            tag = [r.strip() for r in row[7].split(";")]
            tag = tag[0].split(',')
            tag_lstrip = []
            for i in tag:
                tag_lstrip.append(i.lstrip())
            tags_list_test.append(tag_lstrip)
    #x = int(0.8*len(content))
    text_train = desc_train
    text_test = desc_test

    #print len(text_train)
    #print len(text_test)
    start = time.time()

    # create the transform
    vectorizer = TfidfVectorizer()

    # tokenize and build vocab
    data = vectorizer.fit_transform(text_train)

    #Test vector
    vector = vectorizer.transform(text_test)

    
    tfidf_comparisons = []
    for i in range(len(text_test)):
        #print i
        sims = cosine_similarity(vector[i], data)
        tfidf_comparisons.append([sims,i])
    '''
    print sims.shape
    p = tfidf_comparisons[1][0][0]
    print type(p)
    print p

    t = p.argsort()[-5:][::-1]
    print t
    for i in t:
        print p[i]
    print type(sims)
    print type(tfidf_comparisons)
    '''
    #tagsList = set()
    tot_predhas = []
    actualList = []
    a = len(text_test)
    for i in range(a):
        #print i
        tagsList = []
        p_tfidf = tfidf_comparisons[i][0][0]
        srt = p_tfidf.argsort()[-k:][::-1]
        #print "srt printed ::",srt
        for p in range(k):
            tagsList.extend(tags_list_train[srt[p]])
        if '' in tagsList:
            tagsList.remove('')
        if "'#'" in tagsList:
            tagsList.remove("'#'") 
        tot_predhas.append(list(set(tagsList)))
        actualList.append(tags_list_test[i])
        #print "i:: ",i
        #print tagsList
        #print tags_list_test[i]
        #print set(tagsList).intersection(set(tags_list_test[i]))
        #print type(tot_predhas)
        #print type(actualList)
        #print tot_predhas
        #print actualList 

    return tot_predhas,actualList
	
def similarDesc(training_file, testing_file,k):
    #hashtags = [r.strip() for r in row[7].split(";")]
    tot_predhas,actualList = simDesc(training_file,testing_file,k)
    #print res
    return tot_predhas,actualList

#similarDesc("tweets_1lakh_complete.csv","tweets_1lakh_complete.csv",k)




'''
# deleted codes :
--------------------


hash_list = []
t = tags_list_train[srt[p]].split(";")
s = t[0]
hash_list = s.split(',')
tagsList.extend(hash_list)
            
'''

