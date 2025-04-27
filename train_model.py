# train_model.py
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

custom_stopwords = {'a','an','and','are','as','at','be','but','by','for','if','in','into','is','it','no','not','of','on','or','such','that','the','their','then','there','these','they','this','to','was','will','with','you','i'}

def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = [word for word in text.split() if word not in custom_stopwords]
    return ' '.join(words)

df = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\RESUMES\IMDB-Dataset(1) movie.csv")  # replace with actual path
df['cleaned_review'] = df['review'].apply(preprocess)

X_train, _, y_train, _ = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)

with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
