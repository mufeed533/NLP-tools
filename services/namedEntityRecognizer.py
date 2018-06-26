"""
Named entity recognition is mainly used for extracting the noun phrases from the sentences. tokenizer will tokenize sentences and paragraphs. part of speech tagging will assign the part of speeches(noun, verb etc) to the words. chunking will group these words together(for example ger\orge W bush is a noun collectively). Named entity recpgnition can be used to detect entities such as location name from the sentences.
"""
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


class NamedEntityRecognizer():

    def namedEntityRecognizer(self):

        # load training and test data samples
        train_text = state_union.raw("2005-GWBush.txt")
        sample_text = state_union.raw("2006-GWBush.txt")

        # PunktSentenceTokenizer is an unsupervised learning tokenizer which can be trained with custom data
        custom_setence_tokenizer = PunktSentenceTokenizer(train_text)
        tokenized = custom_setence_tokenizer.tokenize(sample_text)

        named_entities = []
        try:
            for i in tokenized[5:]:
                # tokenize the words
                words = nltk.word_tokenize(i)

                # assign part of speech tagging
                tagged = nltk.pos_tag(words)

                #find named entities from the text samples
                namedEnt = nltk.ne_chunk(tagged)
                named_entities.extend([chunk for chunk in namedEnt if hasattr(chunk, 'label')])
            return(named_entities)

        except Exception as e:
            print(str(e))

if __name__ == "__main__":

    # test the functionality
    obj = NamedEntityRecognizer()
    obj.namedEntityRecognizer()
