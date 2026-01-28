#2b
#Building a natural language processing (NLP) model for text classification
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset (News classification)
data = {
    "text": [
        "The team won the championship game yesterday.",
        "The election results will be announced tomorrow.",
        "New smartphone model launched with AI features.",
        "The government passed a new education policy.",
        "The player scored a hat-trick in the football match.",
        "Tech companies are investing in artificial intelligence."
    ],
    "label": ["sports", "politics", "technology", "politics", "sports", "technology"]
}

df = pd.DataFrame(data)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=42
)

# Feature extraction
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualization - Confusion Matrix
labels = ["sports", "politics", "technology"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix - Text Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
