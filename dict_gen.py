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

