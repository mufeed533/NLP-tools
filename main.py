from flask import Flask, request, render_template
from services.sentence_tokenizer import SentenceTokenizer

app: Flask = Flask(__name__)

"""
Method to get input text from user and convert it into sentences
"""
@app.route("/sentenceTokenize", methods=['POST'])
def sentence_tokenize():
    text = request.form.get('text')
    sentence_tokenizer_object = SentenceTokenizer(text)
    tokenized_words = sentence_tokenizer_object.tokenizer()
    return render_template("SentenceTokenizer.html", words = tokenized_words)


"""
Method to get input from user and split teh sentences into words
"""
@app.route("/wordTokenize", methods=['POST'])
def word_tokenize():
    pass


if __name__ == "__main__":
    app.run(debug=True)
