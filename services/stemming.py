from nltk.stem import PorterStemmer
from nltk import word_tokenize

class Stemming:
    text = ""

    def __init__(self, text):
        self.text = text

    def stemmer(self):
        tokenized_words = word_tokenize(self.text)
        stemmer_words = []
        stemmer = PorterStemmer()
        for word in tokenized_words:
            stemmer_words.append(stemmer.stem(word))
        return stemmer_words

if __name__ == "__main__":
    obj = Stemming("I played, playing, plays")
    print(obj.stemmer())
