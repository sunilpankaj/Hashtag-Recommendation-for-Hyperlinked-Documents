from gensim.models import doc2vec
from collections import namedtuple
import pandas as pd 
import csv
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 



#doc1 = ["This is a sentence", "This is another sentence"]

### Transform data (you can add more data preprocessing steps) 

precision = 0.0
recall = 0.0
hitrate = 0.0

list_has = []
def doc2vect(train_file, test_file, k):
	global precision,recall,hitrate
	data = pd.read_csv(train_file)
	docs = []
	analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
	for i in range(data.shape[0]):
		j = i
		words = data['Description'][j].lower().split()
		tags = [i]
		docs.append(analyzedDocument(words, tags))
		  
	#print "hi",  docs
	### Train model (set min_count = 1, if you want the model to work with the provided example data set)

	model = doc2vec.Doc2Vec(docs, size = 100, window = 30, min_count = 1, workers = 4,iter = 500)

	###Get the vectors
	with open(test_file, "r") as file:
		reader = csv.reader(file)
		num = 0
		#next(reader, None)
		tot_predhas = []
		actualList = []
		reader = list(reader) 
		lnt = len(reader)
		for row in range(1,lnt):
			print num
			num +=1
			tweet1 = reader[row]
			#print "dsg ::",tweet
			tweet = tweet1[5]
			hasht = tweet1[7]
			#print hasht
			temp = hasht.split(',')
			#print "length of temp ::",len(temp)
			hashtags = []
			for i in temp:
				hashtags.append(i.lstrip())
			#print hashtags
			#print tweet
			tokens = tweet.split()
			new_vector = model.infer_vector(tokens)
			sims = model.docvecs.most_similar([new_vector])
			pred_hash_tag = []
			l = []
			count = 0
			for i in sims:
				t = i[0]
				count = count + 1;
				l = data['Tags'][t]
			#print type(l)
				if (type(l) == str):
					a = [r.strip() for r in l.split(";")]
					#print len(a)
					a = a[0].split(',')
					for j in a:
						j = j.lstrip()
						pred_hash_tag.append(j)
					#print "predhash ::",pred_hash_tag
				if(count == k):
					break;
			tagsList = pred_hash_tag
			A = list(set(hashtags))
			tot_predhas.append(list(set(tagsList)))
			actualList.append(A)
			'''
			print type(tot_predhas)
			print type(actualList)
			print tot_predhas
			print actualList 
			
			tagsList = pred_hash_tag
			tagsList = set(tagsList)
			emp = []
			A = set(hashtags)
			if (len(A) == 0):
				return emp
			else:
				print A
				print tagsList
				print A.intersection(tagsList)
				intersect = A.intersection(tagsList)
				if ('#Economy' in intersect):
					intersect.remove('#Economy')
				intersect = intersect
				if len(tagsList) != 0:
					#print "precision",len(A.intersection(tagsList))/(len(tagsList)+0.0)
					precision = precision + len(A.intersection(tagsList))/(len(tagsList)+0.0)
				else:
					precision = precision + 0
				#print "recall",len(A.intersection(tagsList))/(len(A)+0.0)
				recall = recall + len(A.intersection(tagsList))/(len(A)+0.0)
				if(len(intersect) > 0):
					hitrate += 1.0 
				#count = count + 1
				print "Total precision is " , precision/num
				print "Total recall is ", recall/num
				print "Total hitrate is ", hitrate/num
				#return list(tagsList) '''
			
			#list_has.extend(pred_hash_tag)
	return tot_predhas,actualList


#listi, j =  doc2vect("training_eco.csv","testing_eco.csv",5)
#print listi , j
#print (Counter(listi))
'''
print len(listi), listi
arr = np.asarray(listi)
#print arr.shape
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(listi)
#X_new_counts = count_vect.transform(listi)
#print X_train_counts.shape
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
print tf_transformer
X_train_tf = tf_transformer.transform(X_train_counts)
print X_train_tf
print (Counter(listi))
#print feature_names[]
a = Counter(listi)
print type(a)
tot = len(a)
maxi =  a.most_common(tot) 
print len(maxi)
freq = []
for i in range(len(maxi)):
	freq.append(maxi[i][1])

count = Counter(freq)
print count

#plotting part 

def simd2v(row):
	dis = row[5]
	res = doc2vect(dis)
	return res
if(__name__ == "__main__"):
	with open("entities4.csv", "r") as file:
		reader = csv.reader(file)
		next(reader, None)
		count  = 0;
		for row in reader:
			count += 1;
			descRes = simd2v(row)
			print Counter(descRes)
			if(count == 1):
				break;
tokens = "new sentence to match".split()

#new_vector = model.infer_vector(tokens)

sims = model.docvecs.most_similar([new_vector]) 

print sims

#print model.most_similar(tags SENT 0)'''
