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



def simContent(train_file,test_file,k):

    content_train = []
    tags_list_train = []
    with open(train_file, "r") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            name = []
            _type = []
            content_train.append(row[6])
            #tag = [r.strip()[1:-1] for r in row[7].split(";")]
            tag = [r.strip() for r in row[7].split(";")]
            #print tag
            tag = tag[0].split(',')
            tag_lstrip = []
            for i in tag:
                tag_lstrip.append(i.lstrip())
            tags_list_train.append(tag_lstrip)
            
    content_test = []
    tags_list_test = []
    with open(test_file, "r") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            name = []
            _type = []
            content_test.append(row[6])
            tag = [r.strip() for r in row[7].split(";")]
            tag = tag[0].split(',')
            tag_lstrip = []
            for i in tag:
                tag_lstrip.append(i.lstrip())
            tags_list_test.append(tag_lstrip)
            #print tag
            
    #x = int(0.8*len(content))
    text_train = content_train
    text_test = content_test

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
    tot_predhas = []
    actualList = []
    for i in range(len(text_test)):
        #print i
        tagsList = []
        p_tfidf = tfidf_comparisons[i][0][0]
        srt = p_tfidf.argsort()[-k:][::-1]
        #print "srt printed ::",srt
        for p in range(k):
            #print i
            tagsList.extend(tags_list_train[srt[p]])
        if '' in tagsList:
            tagsList.remove('')
        if "'#'" in tagsList:
            tagsList.remove("'#'")
        tot_predhas.append(list(set(tagsList)))
        actualList.append(tags_list_test[i])
        #print type(tot_predhas)
        #print type(actualList)
        #print tot_predhas
        #print actualList 
        #print type(tagsList)
        '''print "i ::",i
                                print tags_list_test[i]
                                print tagsList
                                print set(tags_list_test[i]).intersection(set(tagsList))
                                if( i > 10):
                                    break'''
    '''
    for p in range(len(text_test)):
    	tagList = []
        p_tfidf = tfidf_comparisons[p][0][0]
        srt = p_tfidf.argsort()[-5:][::-1]
        #print "srt printed ::",srt
        for i in range(5):
            #print i
            tagsList.extend(tags_list_train[srt[i]])
        if '' in tagsList:
            tagsList.remove('')
        if "'#'" in tagsList:
            tagsListremove("'#'")
        #print type(tagsList)
        print "p ::",p
        print tags_list_test[p]
        print tagsList
        print set(tags_list_test[i]).intersection(set(tagsList))
        if( p > 10):
            break  '''
    return tot_predhas,actualList

    #print "time taken using 20k tweets as training ::",(time.time()-start)
	

def similarCon(trainingfile,testingfile,k):
    #hashtags = [r.strip() for r in row[7].split(";")]
    tot_predhas,actualList = simContent(trainingfile,testingfile,k)
    return tot_predhas,actualList

#similarCon("training_eco.csv","testing_eco.csv")
