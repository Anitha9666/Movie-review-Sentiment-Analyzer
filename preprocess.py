# preprocess.py
import re

custom_stopwords = {'a','an','and','are','as','at','be','but','by','for','if','in','into','is','it','no','not','of','on','or','such','that','the','their','then','there','these','they','this','to','was','will','with','you','i'}

def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = [word for word in text.split() if word not in custom_stopwords]
    return ' '.join(words)
