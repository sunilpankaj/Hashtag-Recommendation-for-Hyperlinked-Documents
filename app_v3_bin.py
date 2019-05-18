import rankfeature
import numpy
import csv
import networkx as nx
import operator
from collections import Counter
import numpy as np
import itertools
import numpy as np
from sklearn import svm, linear_model, cross_validation
from collections import Counter
from sklearn import metrics
import spacy
from sklearn.metrics import confusion_matrix
nlp = spacy.load("en")
import time
#import RWR_test
import LTModel_test
#import domainLink_test
import similarDesc_test
import similarContent_test
import d2v_test
import ranksvm_testing

def transform_pairwise(X, y):
	X_new = []
	y_new = []
	y = np.asarray(y)
	if y.ndim == 1:
		y = np.c_[y, np.ones(y.shape[0])]
	comb = itertools.combinations(range(X.shape[0]), 2)
	for k, (i, j) in enumerate(comb):
		if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
			# skip if same target or different group
			continue
		X_new.append(X[i] - X[j])
		y_new.append(np.sign(y[i, 0] - y[j, 0]))
		# output balanced classes
		if y_new[-1] != (-1) ** k:
			y_new[-1] = - y_new[-1]
			X_new[-1] = - X_new[-1]
	return np.asarray(X_new), np.asarray(y_new).ravel()

class RankSVM(svm.LinearSVC):
	def fit(self, X, y):
		X_trans, y_trans = transform_pairwise(X, y)
		super(RankSVM, self).fit(X_trans, y_trans)
		return self
	def decision_function(self, X):
		return np.dot(X, self.coef_.ravel())
	def predict(self, X):
		if hasattr(self, 'coef_'):
			return np.argsort(np.dot(X, self.coef_.ravel()))
		else:
			raise ValueError("Must call fit() prior to predict()")
	def score(self, X, y):
		X_trans, y_trans = transform_pairwise(X, y)
		return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


X,Y = rankfeature.feature_cre("/home/tdbms06/Code_tweet/testing_tweet.csv")
print " feature matrix completed"
n_samples = X.shape[0]
print n_samples
cv = cross_validation.KFold(n_samples,5,True)
train, test = iter(cv).next()

# print the performance of ranking
start_time3 = time.time()
print " fitting process start "
rank_svm = RankSVM().fit(X[train], Y[train])

print("--- %s svmrank prediction time ---" % (time.time() - start_time3))

ordering = rank_svm.predict(X[test])
print 'Performance of ranking ', rank_svm.score(X[test], Y[test])

#######################################################################

#							TESTING PART  							  #

#######################################################################

precision = []
recall = []
hitrate = []

if(__name__ == "__main__"):
	with open("/home/tdbms06/Code_tweet/testing_tweet.csv", "r") as file:
		#global precision,recall,hitrate
		reader = csv.reader(file)
		rows = list(reader)
		# total number of posts = 501
		#test_index = range(450,460)
		test_index = range(1,len(rows))
		for j in test_index:
			#print "j :: ",j
			row = rows[j]
			#list of actual hashtags
			act_tweet = []
			kact = row[7];
			kact = ''.join(kact).replace(' ',"").split(';')
			kact = kact[0].split(',')
			for i in kact:
				i = i.lstrip('#')
				i = i.lower()
				act_tweet.append(i.lstrip())
			#list of predicted hashtags
			X_t,Y_t,pred_hash = ranksvm_testing.feature_cre(row)

			ordering_test = rank_svm.predict(X_t)
			k = 5
			p = ordering_test.argsort()[-k:][::-1]
			#print p 
			kpredict = []
			for i in p:
				t = pred_hash[i].lstrip('#')
				t = t.lower()
				kpredict.append(t)
			kpredict = ','.join(kpredict).replace(' ',"").split(',')
			
			intersect = list(set(kpredict).intersection(set(act_tweet)))
			print kpredict
			print act_tweet
			print intersect
			#print " tweet number :: ",j
			#print "p :: ",p
			#print "g :: ",g 
			prec = float(len(intersect))/ float(len(p))
			precision.append(prec)
			#print prec

			rec= float(len(intersect))/ float(len(act_tweet))
			recall.append(rec)
			#print rec 

			if len(intersect) >= 1:
				h = 1
			else:
				h = 0
			hitrate.append(h)
			print "Avg precision :: ",sum(precision)/float(len(precision)+1)
			print "Avg recall:: ",sum(recall)/float(len(recall)+1)
			print "Avg hitrate:: ",sum(hitrate)/float(len(hitrate)+1)
			#print h
print "Avg precision :: ",sum(precision)/float(len(precision))
print "Avg recall:: ",sum(recall)/float(len(recall))
print "Avg hitrate:: ",sum(hitrate)/float(len(hitrate))

			




















