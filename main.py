from flask import Flask, request, render_template
from services.sentence_tokenizer import SentenceTokenizer
from services.word_tokenizer import WordTokenizer
from services.StopWords import StopWords
from services.stemming import Stemming
from services.partOfSpeechTagging import PartOfSpeechTagging
from services.namedEntityRecognizer import NamedEntityRecognizer
from services.lemmatizer import Lemmatizer
from services.wordNet import WordDetails

app: Flask = Flask(__name__)

"""
Controller to convert the string into sentences
"""
@app.route("/sentenceTokenize", methods=['POST'])
def sentence_tokenize():
    text = request.form.get('text')
    sentence_tokenizer_object = SentenceTokenizer(text)
    tokenized_sentences = sentence_tokenizer_object.tokenizer()
    return render_template("SentenceTokenizer.html", sentences = tokenized_sentences)


"""
Controller to split the sentences into words
"""
@app.route("/wordTokenize", methods=['POST'])
def word_tokenize():
    text = request.form.get('text')
    word_tokenizer_object = WordTokenizer(text)
    tokenized_words = word_tokenizer_object.tokenizer()
    return render_template("WordTokenizer.html", words = tokenized_words)


"""
Controller to remove all the stop words from the string
"""
@app.route("/stopWordRemover", methods=['POST'])
def stop_word_remover():
    text = request.form.get('text')
    stop_words_object = StopWords(text)
    filtered_text = stop_words_object.StopWordRemove()
    return render_template("StopWords.html", sentence = filtered_text)

"""
Controller to get the root words from a sentence
"""
@app.route("/stemmer", methods = ['POST'])
def stemmer():
    text = request.form.get('text')
    stemming_object = Stemming(text)
    stemmer_words = stemming_object.stemmer()
    return render_template("stemmer.html", words = stemmer_words)


"""
Controller for applying part fo speech tagging for sample text
"""
@app.route("/posTagging", methods = ['GET', 'POST'])
def pos_tagging():
    pos_object = PartOfSpeechTagging()
    all_tags = pos_object.partOfSpeechTagging()
    return render_template("partOfSpeechtagging.html" , all_tags = all_tags)


"""
Controller for identifying named entities from sample text
"""
@app.route("/namedEntityRecognizer", methods = ['GET'])
def named_entity_recognize():
    named_enitoty_recognizer_object = NamedEntityRecognizer()
    named_entities = named_enitoty_recognizer_object.namedEntityRecognizer()
    return render_template("namedEntityRecognizer.html", named_entities = named_entities)


"""
Controller for finding the useful route word for a word
"""
@app.route("/lemmatize", methods = ['POST'])
def lemmatizer():
    text =  request.form.get("text")
    obj = Lemmatizer(text)
    lemmatized_word = [text,obj.wordLemmatize()]
    return render_template("lemmatizer.html", lemmatized_word = lemmatized_word)


"""
Controller to get details of the word such as definition, synonyms, antonyms and examples
"""
@app.route("/details", methods = ['POST'])
def wordDetails():
    text =  request.form.get("text")
    obj = WordDetails(text)
    details = []
    details.append(text)
    details.append(obj.definition())
    details.append(obj.synonyms())
    details.append(obj.antonyms())
    details.append(obj.examples())
    print(details)
    return render_template("wordDetails.html",details = details)

if __name__ == "__main__":
    app.run(debug=True)
