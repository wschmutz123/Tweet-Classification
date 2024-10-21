# Tweet-Classification

This repository looks at Tweets and classifies them as positive, negative, or neutral

The entire program is written in Python (libraries used: scikit-learn, pandas, numpy)

Some concepts Used:

  - TFIDF Vectorizer: Vectorize the tweet into an array of numerical values making it easier to test
  - KFold: Uses Cross Validation to evaluate the performance of my model by splitting the result in K subsets
  - SVC: Used support vector machines which uses clustering to seperate the classes into a feature space and is especially useful when we are dealing with high-demensional datasets
