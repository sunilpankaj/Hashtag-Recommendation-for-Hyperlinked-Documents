Hashtag recommendation method has two phases: candidate hashtag selection and hashtag ranking
Candidate hashtag selection: 
1. Hashtags from Similar Description(similarContent.py)
2. Hashtags from Similar Content(similarDesc.py)
3. Language Translation Model (LTModel.py)
4. Random Walk with Restart (RWR) model (RWR.py)
5. doc2vec(d2v.py)

hashtag ranking: Recommendation by Learning to Rank
Used Pairwise learning to rank and rank svm for ranking the hashtags.






