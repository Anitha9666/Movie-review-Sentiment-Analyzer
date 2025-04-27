# app.py
from flask import Flask, render_template, request
import pickle
from preprocess import preprocess

app = Flask(__name__)

# Load model
with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        review = request.form['review']
        cleaned = preprocess(review)
        prediction = model.predict([cleaned])[0]
        
        if prediction.lower() == "positive":
            sentiment = "Positive 😊"
        else:
            sentiment = "Negative 😞"

    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)

