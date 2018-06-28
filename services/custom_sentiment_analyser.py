"""
A custom sentiment analysis. algorithm is trained with our data. code is incomplete.
"""
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
import os
from nltk.tokenize import word_tokenize

file_path = os.path.dirname(os.getcwd())

# load negative and positive data. I used errors = 'ignore' because there is a chance of unicodEerror
with open(file_path+"/datasets/negative.txt","r", errors='ignore') as fd:
    negative_text = fd.read()
with open(file_path+"/datasets/positive.txt","r", errors='ignore') as fd:
    positive_text = fd.read()

documents = []

# label each line as negative or positive and add to documents
for review in negative_text.split('\n'):
    documents.append((review, "neg"))
for review in positive_text.split('\n'):
    documents.append((review, "pos"))


# split positive_text and negative_text word by word
positive_words = word_tokenize(positive_text)
negative_words = word_tokenize(negative_text)

all_words = []

# add every words (negative and positve) to a new list with lower case
for word in positive_words:
    all_words.append(word.lower())
for word in negative_words:
    all_words.append(word.lower())

# take only 5000 most frequent words assuming remaining words are negligible
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = []
for (review, category) in documents:
    featuresets.append((find_features(review),category))

print(featuresets)
