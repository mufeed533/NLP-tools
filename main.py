from flask import Flask, request, render_template
from services.sentence_tokenizer import SentenceTokenizer
from services.word_tokenizer import WordTokenizer

app: Flask = Flask(__name__)

"""
Method to get input text from user and convert it into sentences
"""
@app.route("/sentenceTokenize", methods=['POST'])
def sentence_tokenize():
    text = request.form.get('text')
    sentence_tokenizer_object = SentenceTokenizer(text)
    tokenized_sentences = sentence_tokenizer_object.tokenizer()
    return render_template("SentenceTokenizer.html", sentences = tokenized_sentences)


"""
Method to get input from user and split teh sentences into words
"""
@app.route("/wordTokenize", methods=['POST'])
def word_tokenize():
    text = request.form.get('text')
    word_tokenizer_object = WordTokenizer(text)
    tokenized_words = word_tokenizer_object.tokenizer()
    return render_template("WordTokenizer.html", words = tokenized_words)



if __name__ == "__main__":
    app.run(debug=True)
