import matplotlib.pyplot as plt
import csv
import numpy as np
import operator

sumTags = 0
docCount = 0
hTagFreq = dict()
tagFreqDict = dict()
lenDict = dict()

with open("politics_refined.csv", "r") as file:
    reader = csv.reader(file)
    next(reader, None)

    for row in reader:
    	rowTags = [r.strip() for r in row[7].split(";")]
    	sumTags += len(rowTags)
    	if(len(rowTags) in lenDict):
    		lenDict[len(rowTags)] += 1
    	else:
    		lenDict[len(rowTags)] = 1

    	for tag in rowTags:
    		if(tag in hTagFreq):
    			hTagFreq[tag] += 1
    		else:
    			hTagFreq[tag] = 1
    	docCount += 1

for u,v in hTagFreq.items():
	if(v in tagFreqDict):
		tagFreqDict[v] += 1
	else:
		tagFreqDict[v] = 1

#bar graph
sumGTE15 = 0
x_str = []
y = []
for key in lenDict.keys():
    if(key > 15):
        sumGTE15 += lenDict[key]
    else:
        x_str.append(str(key))
        y.append(lenDict[key])
x_str.append(">15")
y.append(sumGTE15)
y = [float(val) / docCount for val in y]

plt.bar(range(1, len(x_str) + 1), y)
plt.xticks(range(1, len(x_str) + 1), x_str)
plt.xlabel("Number of hashtags per post")
plt.ylabel("Percent of posts(%)")
plt.gca().yaxis.grid(True)
plt.show()

#Scatter plot
plt.scatter(list(tagFreqDict.keys()), list(tagFreqDict.values()), marker='x')
plt.xlabel("Hashtag frequency")
plt.ylabel("Number of hashtags")
plt.grid()
plt.ylim(ymin=-10, ymax=100)
plt.xlim(xmax=1000)
plt.show()