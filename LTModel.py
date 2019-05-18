import csv
import operator
import matplotlib.pyplot as plt
import operator
import time

#Count the cooccurences of an entity and hashtag
entTagCount = dict()

#Entity and tags count
entityCount = dict()


#P(H/E)
probHE = dict()


def preProcess(fil):
	with open(fil, "r") as file:
		reader = csv.reader(file)
		next(reader, None)
		for row in reader:
			rowTags = []
			t = [r.strip() for r in row[7].split(";")]
			p = t[0].split(',')
			for i in p:
				rowTags.append(i.lstrip())

			if("" in rowTags):
				rowTags.remove("")
			if "'#'" in rowTags:
				rowTags.remove("'#'")

			#print rowTags
			#entities = [r.strip().split('#$#')[0] for r in row[8].split(";")]
			ent = [r.strip().split('#$#')[0] for r in row[8].split(";")]
			#print "ent ::",ent
			#ent1 = ent[0].split(',')
			entities = []
			for i in ent:
				entities.append(i.lstrip())
			#print entities
			if("" in entities):
				entities.remove("")
			if "'#'" in entities:
				entities.remove("'#'")
			#print entities
			#Entity count
			for entity in entities:
				if(entity in entityCount):
					entityCount[entity] += 1
				else:
					entityCount[entity] = 1

		#Entity tag count
			for entity in entities:
				for tag in rowTags:
					if((entity, tag) in entTagCount):
						entTagCount[(entity, tag)] += 1
					else:
						entTagCount[(entity, tag)] = 1
	#P(H/E)
	for u, v in entTagCount.items():
		probHE[(u[1], u[0])] = float(v) / entityCount[u[0]]


preProcess("training_eco.csv")

def LT(train_file, test_file,K):
	#func(train_file)
	tot_predhas = []
	actualList = []
	final_list = []
	with open(test_file, "r") as file:
		reader = csv.reader(file)
		next(reader, None)
		count = 0;
		for row in reader:
			if(count % 250 == 0 ):
				print count
			entities = [r.strip() for r in row[8].split(";")]
			if("" in entities):
				entities.remove("")
			if "'#'" in entities:
				entities.remove("'#'")

			entities = [r.strip().split('#$#')[0] for r in entities]
			#print entities
			rowTags = set()
			for u, v in entTagCount.items():
				if(u[0] in entities):
					rowTags.add(u[1])
			rowTags = list(rowTags)
			
			topK = []
			for r in rowTags:
				k = 0
				for e in entities:
					if (r,e) in probHE:
						k = k + probHE[(r,e)]
				topK.append((r,k))
			
			top = sorted(range(len(topK)), key=lambda i: topK[i][1], reverse=True)[:K]

			result = []
			for i in range(len(top)):
				result.append(topK[top[i]][0])
			#final_list.extend(result)
			#finding actual hashtags
			tag = [r.strip() for r in row[7].split(";")]
			tag = tag[0].split(',')
			tag_lstrip = []
			for i in tag:
				tag_lstrip.append(i.lstrip())
			
			A = set(tag_lstrip)
			result = set(result)
			intersect = A.intersection(result)
			'''print "count :: ",count
												print A
												print result
												print intersect
												print "        "'''
			tot_predhas.append(list(set(result)))
			actualList.append(list(set(tag_lstrip)))
			count += 1
			#print final_list
			#print type(tot_predhas)
			#print type(actualList)
			#print tot_predhas
			#print actualList 
	return tot_predhas,actualList
	


#start = time.time()
#a = LT("training_eco.csv","testing_eco.csv",5)
#print a
#print "time tken for LTModel (20k posts ):: ",(time.time()-start)
#print a
