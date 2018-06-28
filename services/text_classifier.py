import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import pickle
from nltk.classify import ClassifierI
from statistics import mode


"""
A class for getting the votes from all the classifiers
"""
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        conf = votes.count(mode(votes)) / len(votes)
        return conf


documents = []
# there are two categories in the movie_reviews data. Negative and Positive
for categories in movie_reviews.categories():
    # fileIds are basically file names of positive and negative emotions. For example: neg/cv030_22893.txt, pos/cv298_23111.txt
    for fileid in movie_reviews.fileids(categories):
        documents.append((list(movie_reviews.words(fileid)),categories))

random.shuffle(documents)
all_words = []
for words in movie_reviews.words():
    all_words.append(words.lower())


all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featuresets = [(find_features(rev), category)for (rev, category) in documents]
training_set = featuresets[:1900]
test_set = featuresets[1900:]


# example to train and save the model. I commented it because I already saved the model to save time of retraining
# classifier = nltk.NaiveBayesClassifier.train(training_set)
# save_classifier = open("NaiveBayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()


# original NLTK Naive Bayes classifier
saved_classifier = open("saved_classifiers/NaiveBayes.pickle","rb")
classifier = pickle.load(saved_classifier)
saved_classifier.close()
classifier.show_most_informative_features(15)
print("Origianal naive NaiveBayes accuracy percent : ",(nltk.classify.accuracy(classifier, test_set))*100)


# MNBclassifier
saved_classifier = open("saved_classifiers/MNB.pickle","rb")
MNB_classifier = pickle.load(saved_classifier)
saved_classifier.close()
print("MNBclassifier accuracy percent : ",(nltk.classify.accuracy(MNB_classifier, test_set))*100)


# GaussianNB classifier
# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier accuracy percent : ",(nltk.classify.accuracy(GaussianNB_classifier, test_set))*100)


# BernoulliNB classifier
saved_classifier = open("saved_classifiers/BernoulliNB.pickle","rb")
BernoulliNB_classifier = pickle.load(saved_classifier)
saved_classifier.close()
print("BernoulliNB_classifier accuracy percent : ",(nltk.classify.accuracy(BernoulliNB_classifier, test_set))*100)


# LogisticRegression classifier
saved_classifier = open("saved_classifiers/logisiticRegression.pickle","rb")
LogisticRegression_classifier = pickle.load(saved_classifier)
saved_classifier.close()
print("LogisticRegression_classifier accuracy percent : ",(nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)


# SGDClassifier classifier
saved_classifier = open("saved_classifiers/SGDClassifier.pickle","rb")
SGDClassifier_classifier = pickle.load(saved_classifier)
saved_classifier.close()
print("SGDClassifier_classifier accuracy percent : ",(nltk.classify.accuracy(SGDClassifier_classifier, test_set))*100)


# SVC classifier
saved_classifier = open("saved_classifiers/SVC_classifier.pickle","rb")
SVC_classifier = pickle.load(saved_classifier)
saved_classifier.close()
print("SVC_classifier accuracy percent : ",(nltk.classify.accuracy(SVC_classifier, test_set))*100)


# LinearSVC classifier
saved_classifier = open("saved_classifiers/LinearSVC_classifier.pickle","rb")
LinearSVC_classifier = pickle.load(saved_classifier)
saved_classifier.close()
print("LinearSVC_classifier accuracy percent : ",(nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)


# NuSVC classifier
saved_classifier = open("saved_classifiers/NuSVC_classifier.pickle","rb")
NuSVC_classifier = pickle.load(saved_classifier)
saved_classifier.close()
print("NuSVC_classifier accuracy percent : ",(nltk.classify.accuracy(NuSVC_classifier, test_set))*100)


# voting section
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SVC_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print("voted classifer accuracy percent : ", (nltk.classify.accuracy(voted_classifier, test_set)) * 100)
print("testing sample : ")
for i in test_set[0][0].keys():
    print(i,end = " ")
print("\nclassification : ", voted_classifier.classify(test_set[0][0]), "confidence :", voted_classifier.confidence(test_set[0][0]))
