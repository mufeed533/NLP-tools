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

# from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
#
#
# def extract_entities(text):
# 	entities = []
# 	for sentence in sent_tokenize(text):
# 	    chunks = ne_chunk(pos_tag(word_tokenize(sentence)))
# 	    entities.extend([chunk for chunk in chunks if hasattr(chunk, 'label')])
# 	return entities
#
#
# if __name__ == '__main__':
# 	text = """
# A multi-agency manhunt is under way across several states and Mexico after
# police say the former Los Angeles police officer suspected in the murders of a
# college basketball coach and her fiancÃ© last weekend is following through on
# his vow to kill police officers after he opened fire Wednesday night on three
# police officers, killing one.
# "In this case, we're his target," Sgt. Rudy Lopez from the Corona Police
# Department said at a press conference.
# The suspect has been identified as Christopher Jordan Dorner, 33, and he is
# considered extremely dangerous and armed with multiple weapons, authorities
# say. The killings appear to be retribution for his 2009 termination from the
#  Los Angeles Police Department for making false statements, authorities say.
# Dorner posted an online manifesto that warned, "I will bring unconventional
# and asymmetrical warfare to those in LAPD uniform whether on or off duty."
# """
# 	print(extract_entities(text))
