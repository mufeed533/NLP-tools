from nltk.corpus import stopwords

class StopWords:

    def __init__(self, text):
        self.text = text
        self.filtered_words = []
        self.stop_words = set(stopwords.words("english"))

    def StopWordRemove(self):
        for words in self.text.split():
            if words not in self.stop_words:
                self.filtered_words.append(words)
        return " ".join(self.filtered_words)

if __name__ == "__main__":
    obj = StopWords("Amazon Lex is a service for building conversational interfaces into any application using voice and text. Amazon Lex provides the advanced deep learning functionalities of automatic speech recognition (ASR) for converting speech to text, and natural language understanding (NLU) to recognize the intent of the text, to enable you to build applications with highly engaging user experiences and lifelike conversational interactions. With Amazon Lex, the same deep learning technologies that power Amazon Alexa are now available to any developer, enabling you to quickly and easily build sophisticated, natural language, conversational bots (“chatbots”).")
    a = obj.StopWordRemove()
    print(a)
