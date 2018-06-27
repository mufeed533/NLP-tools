import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import pickle

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

# classifier = nltk.NaiveBayesClassifier.train(training_set)
# save_classifier = open("NaiveBayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()


# original NLTK Naive Bayes classifier
saved_classifier = open("NaiveBayes.pickle","rb")
classifier = pickle.load(saved_classifier)
saved_classifier.close()
classifier.show_most_informative_features(30)
print("Origianal naive NaiveBayes accuracy percent : ",(nltk.classify.accuracy(classifier, test_set))*100)

# MNBclassifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNBclassifier accuracy percent : ",(nltk.classify.accuracy(MNB_classifier, test_set))*100)

# GaussianNB classifier
# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier accuracy percent : ",(nltk.classify.accuracy(GaussianNB_classifier, test_set))*100)

# BernoulliNB classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent : ",(nltk.classify.accuracy(BernoulliNB_classifier, test_set))*100)

# LogisticRegression classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent : ",(nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)


# SGDClassifier classifier
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent : ",(nltk.classify.accuracy(SGDClassifier_classifier, test_set))*100)

# SVC classifier
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent : ",(nltk.classify.accuracy(SVC_classifier, test_set))*100)

# LinearSVC classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent : ",(nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)


# NuSVC classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent : ",(nltk.classify.accuracy(NuSVC_classifier, test_set))*100)
