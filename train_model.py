import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['text'] = df['text'].str.lower()

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, 'spam_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
