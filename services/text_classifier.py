import nltk
import random
from nltk.corpus import movie_reviews

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

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("nltk classification accuracy percent : ",(nltk.classify.accuracy(classifier, test_set))*100)
classifier.show_most_informative_features(30)
