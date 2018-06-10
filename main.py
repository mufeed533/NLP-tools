from flask import Flask, request, render_template
from services.sentence_tokenizer import SentenceTokenizer
from services.word_tokenizer import WordTokenizer
from services.StopWords import StopWords
from services.stemming import Stemming
from services.partOfSpeechTagging import PartOfSpeechTagging

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


if __name__ == "__main__":
    app.run(debug=True)
