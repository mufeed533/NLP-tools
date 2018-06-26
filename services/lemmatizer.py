"""
Lemmatizing is similar to stemming. But stemming sometimes lead to non existing words. To get the actual word we can use lemmatization.
"""
from nltk.stem import WordNetLemmatizer

class Lemmatizer():
    text = ""
    def __init__(self, text):
        self.text = text

    def wordLemmatize(self):
        lemmatizer = WordNetLemmatizer()
        print(lemmatizer.lemmatize(self.text)) # lemmatize() has an opetional parameter called pos = 'n' we can mention the pos there. Default pos is 'n'

if __name__ == "__main__":
    obj = Lemmatizer("cats")
    obj.wordLemmatize()
