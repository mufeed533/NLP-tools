from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
import nltk


class PartOfSpeechTagging:

    def partOfSpeechTagging(self):
        train_text = state_union.raw("2005-GWBush.txt")
        sample_text = state_union.raw("2006-GWBush.txt")

        # train the PunktSentenceTokenizer
        cust_Sentence_tokenizer = PunktSentenceTokenizer(train_text)
        tokenized = cust_Sentence_tokenizer.tokenize(sample_text)

        try:
            all_tags = []
            for i in tokenized:
                words = nltk.word_tokenize(i)
                tags = nltk.pos_tag(words)
                all_tags.append(tags)
            return all_tags

        except Exception as e:
            print(str(e))

if __name__ == "__main__":
    obj = PartOfSpeechTagging()
    obj.partOfSpeechTagging()
