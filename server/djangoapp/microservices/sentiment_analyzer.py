from flask import Flask, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# sicherstellen, dass die Vader-Daten verfügbar sind
nltk.download('vader_lexicon')

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

@app.route("/analyze/<text>")
def analyze(text):
    scores = analyzer.polarity_scores(text)
    return jsonify(scores)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
