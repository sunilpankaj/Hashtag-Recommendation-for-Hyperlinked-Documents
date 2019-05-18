"""
	Program: Implementation of DICT GENERATION
"""

import csv
import networkx as nx
import operator

#Count the cooccurences of an entity and hashtag
entTagCount = dict()

#Pairwise tags cooccurence count
coOccCount = dict()

#Entity and tags count
entityCount = dict()
tagCount = dict()

#P(E/H)
probEH = dict()

#P(H/E)
probHE = dict()

#P(hj/hk)
pCoOcc = dict()

def preProcess():
	with open("entities4.csv", "r") as file:
		reader = csv.reader(file)
		next(reader, None)

		for row in reader:
			rowTags = [r.strip() for r in row[7].split(";")]
			print(rowTags)
			if("" in rowTags):
				rowTags.remove("")
			if "'#'" in rowTags:
				rowTags.remove("'#'")

			entities = [r.strip().split('#$#')[0] for r in row[8].split(";")]
			if("" in entities):
				entities.remove("")
			if "'#'" in entities:
				entities.remove("'#'")

			#Entity count
			for entity in entities:
				if(entity in entityCount):
					entityCount[entity] += 1
				else:
					entityCount[entity] = 1

			#Tag count
			for tag in rowTags:
				if(tag in tagCount):
					tagCount[tag] += 1
				else:
					tagCount[tag] = 1

			#Entity tag count
			for entity in entities:
				for tag in rowTags:
					if((entity, tag) in entTagCount):
						entTagCount[(entity, tag)] += 1
					else:
						entTagCount[(entity, tag)] = 1

def dict_gen():
	preProcess()

dict_gen()
print tagCount

import itertools
import numpy as np
from sklearn import svm, linear_model, cross_validation

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


from collections import Counter
import numpy as np
X = np.ndarray((len(list_predict),5))
print X.shape
Y = np.ndarray((len(list_predict),2))

if __name__ == '__main__':
    # as showcase, we will create some non-linear data
    # and print the performance of ranking vs linear regression

    for i in range(len(list_predict)):
        X[i][0] = 1;
        for j in range(1,5):
            X[i][j] = np.random.randint(2,size=1)[0]

        if(list_predict[i] in tagCount):
            #index = hashtag_dict.index(list_predict[i])
            #unique_words = set(term_dict[index])
            num_term = tagCount[list_predict[i]]
            #print num_term
            label = (int)(num_term / 5) + 1
            if(label > 10):
                Y[i][0] = 10
            else:
                #Y[i][0] = label
                Y[i][0] = label
            Y[i][1] = 0
        else:
            Y[i][0] = 0
            Y[i][1] = 0
            
    #X = np.random.randn(n_samples, n_features)
    #y = np.random.randint(2, size=(n_samples, 1))
    print X,Y
    n_samples = len(list_predict)
    cv = cross_validation.KFold(n_samples, 5)
    train, test = iter(cv).next()
    # print the performance of ranking
    rank_svm = RankSVM().fit(X[train], Y[train])
    print rank_svm.predict(X[test])
    print 'Performance of ranking ', rank_svm.score(X[test], Y[test])
  

