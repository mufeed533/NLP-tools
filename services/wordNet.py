"""
The wordnet is a module which is a part of nltk. The wordnet module provides the basic features of words such as definitions, examples, antonyms, synonyms etc.
"""
from nltk.corpus import wordnet

class WordDetails:
    word = ""
    def __init__(self, word):
        self.word = word
        # get synonyms of the word
        self.syns = wordnet.synsets(word)


    def definition(self):
        return(self.syns[0].definition())


    def synonyms(self):
        synonyms = set()
        for i in self.syns:
            synonyms.add(i.lemmas()[0].name())

        # if there are no synonyms add the text no synonyms
        if len(synonyms) == 0:
            antonyms.add("no synonyms")
        return(synonyms)


    def antonyms(self):
        antonyms = set()
        for syn in self.syns:
            for i in syn.lemmas():
                if(i.antonyms()):
                    antonyms.add(i.antonyms()[0].name())

        # if there are no antonyms add the text no antonyms
        if len(antonyms) == 0:
            antonyms.add("no antonyms")
        return(antonyms)

    def examples(self):
        return(self.syns[0].examples())


if __name__ == '__main__':

    # testing
    obj = WordDetails("program")
    obj.synonyms()
    obj.antonyms()
    obj.exmaples()
