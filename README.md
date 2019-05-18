use crawler.py to scrap news data from moneycontrol.com site.
Sample of data: testing_eco.csv

Hashtag recommendation method has two phases: candidate hashtag selection and hashtag ranking
Candidate hashtag selection: 
1. Hashtags from Similar Description(sim_content.py)
2. Hashtags from Similar Content(sim_desc.py)
3. Language Translation Model (LTModel.py)
4. Random Walk with Restart (RWR) model (RWR.py)
5. doc2vec(d2v.py)

hashtag ranking: Recommendation by Learning to Rank
Used Pairwise learning to rank and rank svm for ranking the hashtags.

app_v3_bin.py : pairwise learning and ranksvm

Evaluation:
Precision: https://github.com/sunilpankaj/HashtagRecommendation/blob/master/precision.jpg











