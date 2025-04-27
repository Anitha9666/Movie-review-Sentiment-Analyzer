**Movie Review Sentiment Analyzer**
**Description**
The Movie Review Sentiment Analyzer is a web application that analyzes the sentiment of movie reviews. It uses machine learning models to predict whether a given review is positive or negative based on the content. This project compares three different Naive Bayes models (Gaussian, Multinomial, Bernoulli) and evaluates their performance using key metrics like accuracy, precision, recall, and F1-score.

The application provides a Flask-based web interface where users can input their reviews and receive sentiment predictions in real-time.

**Features**
Sentiment prediction for movie reviews: Positive or Negative

Three Naive Bayes models for comparison: Gaussian, Multinomial, Bernoulli

Performance metrics: Accuracy, Precision, Recall, and F1-score

User-friendly Flask web interface for easy review input and result display

Text preprocessing pipeline: Tokenization, Vectorization, and Cleaning

**Technologies Used**
Python: Main programming language for developing the models and application
Flask: Web framework for the frontend
scikit-learn: For machine learning models
pickle: For saving and loading models
HTML/CSS: For creating the web interface
Matplotlib/Seaborn: For visualizing metrics (if needed)

Installation
Prerequisites
Make sure you have Python 3.x installed on your system. Additionally, you will need the following Python libraries:
Flask
scikit-learn
pickle
matplotlib (optional, for graphs)
pandas
numpy
You can install these dependencies using pip:
pip install flask scikit-learn matplotlib pandas numpy
Setup
Clone this repository to your local machine:


git clone https://github.com/yourusername/movie-review-sentiment-analyzer.git
cd movie-review-sentiment-analyzer
Place the saved model (sentiment_model.pkl) inside the model/ directory.
Run the Flask application:
python app.py
Open your browser and navigate to http://127.0.0.1:5000/ to access the application.

Usage
Enter a movie review in the text input box.

Click Submit to see whether the sentiment of the review is predicted as Positive or Negative.

The model used for prediction will be the one with the best performance, based on metrics evaluated during training.

Evaluation Metrics
The project evaluates the performance of three Naive Bayes models:

Gaussian Naive Bayes: Accuracy = 0.7843

Multinomial Naive Bayes: Accuracy = 0.831

Bernoulli Naive Bayes: Accuracy = 0.8386

Based on these metrics, Bernoulli Naive Bayes was found to be the most accurate for this task.

How It Works
Text Preprocessing: The review text is cleaned by removing stopwords, punctuation, and applying tokenization.

Feature Extraction: A Bag of Words (BOW) model is used to convert the text data into a matrix of token counts.

Model Training: The three Naive Bayes models are trained on a labeled dataset of movie reviews.

Prediction: Upon receiving a user input, the review is preprocessed and then passed through the selected model to predict sentiment.

Contributing
Feel free to fork this repository, make improvements, or report any issues.

Fork the repository
Create a feature branch (git checkout -b feature-branch)
Commit your changes (git commit -m 'Add feature')
Push to the branch (git push origin feature-branch)
Open a Pull Request
