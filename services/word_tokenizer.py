from nltk import word_tokenize

class WordTokenizer:
    text = ""

    def __init__(self, text):
        self.text = text

    def tokenizer(self):
        return word_tokenize(self.text)
