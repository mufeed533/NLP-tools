from nltk import sent_tokenize

class SentenceTokenizer:
    text = ""

    def __init__(self, text):
        self.text = text

    def tokenizer(self):
        return sent_tokenize(self.text)
