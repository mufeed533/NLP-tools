from flask import Flask, request, render_template
from services.sentence_tokenizer import SentenceTokenizer

app: Flask = Flask(__name__)


@app.route("/home")
def home():
    nltk.download()
    return "hai"


@app.route("/test", methods=['POST', 'GET'])
def test():
    return "%s" % request.form.get('name')


@app.route("/sentenceTokenize", methods=['POST'])
def sentence_tokenize():
    text = request.form.get('text')
    print(text)
    sentence_tokenizer_object = SentenceTokenizer(text)
    tokenized_words = sentence_tokenizer_object.tokenizer()
    return render_template("SentenceTokenizer.html", words = tokenized_words)


@app.route("/wordTokenize", methods=['POST'])
def word_tokenize():
    pass


if __name__ == "__main__":
    app.run(debug=True)
