import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from preprocess import preprocess

# Load model
with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset (replace with your dataset path)
df = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\RESUMES\IMDB-Dataset(1) movie.csv")  # Modify this path to your actual dataset file
X = df['review']  # Assuming 'review' is the column with text data
y = df['sentiment']  # Assuming 'sentiment' is the column with labels (positive/negative)

# Preprocess the data
X_cleaned = X.apply(preprocess)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.3, random_state=42)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive')
recall = recall_score(y_test, y_pred, pos_label='positive')
f1 = f1_score(y_test, y_pred, pos_label='positive')

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
