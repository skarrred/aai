# Practical no: 2
### A) Building a natural language processing (NLP) model for sentimental analysis.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset
data = {
    "text": [
        "I love this phone, it's awesome!",
        "The movie was terrible and boring.",
        "I am happy with the product.",
        "Worst service ever, I am disappointed.",
        "Excellent experience, very satisfied.",
        "The food was bad and cold."
    ],
    "label": ["positive", "negative", "positive", "negative", "positive", "negative"]
}

df = pd.DataFrame(data)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=42
)

# Feature extraction
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualization - Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["positive", "negative"],
    yticklabels=["positive", "negative"]
)

plt.title("Confusion Matrix - Sentiment Analysis")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
