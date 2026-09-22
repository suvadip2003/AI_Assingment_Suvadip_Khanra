import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


print("=" * 60)
print("CUSTOMER SUPPORT TICKET CLASSIFICATION")
print("=" * 60)

df = pd.read_csv("data/tickets.csv")

print("\nDataset loaded successfully!")
print("Total records:", len(df))


df["ticket_description"] = (
    df["ticket_description"]
    .fillna("")
    .astype(str)
)

# Convert text to lowercase
df["ticket_description"] = (
    df["ticket_description"].str.lower()
)

# Input
X = df["ticket_description"]

# Target
y = df["category"]


print("\nCategories:")
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))


print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    "TF-IDF training feature shape:",
    X_train_tfidf.shape
)

print(
    "TF-IDF testing feature shape:",
    X_test_tfidf.shape
)


print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)


print("Model training completed!")


y_pred = model.predict(
    X_test_tfidf
)


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)


os.makedirs(
    "outputs",
    exist_ok=True
)


plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title(
    "Confusion Matrix - Ticket Classification"
)

plt.xlabel(
    "Predicted Category"
)

plt.ylabel(
    "Actual Category"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    "outputs/confusion_matrix.png"
)


os.makedirs(
    "model",
    exist_ok=True
)


joblib.dump(
    model,
    "model/ticket_classifier.pkl"
)

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)


print("\n" + "=" * 60)
print("MODEL FILES SAVED")
print("=" * 60)

print(
    "model/ticket_classifier.pkl"
)

print(
    "model/tfidf_vectorizer.pkl"
)


print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

sample_tickets = [
    "I forgot my password and cannot login",
    "The application crashes when I save data",
    "I need the monthly sales report",
    "Please change my registered phone number",
    "The application is very slow today"
]


sample_features = vectorizer.transform(
    sample_tickets
)

sample_predictions = model.predict(
    sample_features
)


for ticket, prediction in zip(
    sample_tickets,
    sample_predictions
):

    print("\nTicket:", ticket)
    print("Prediction:", prediction)


print("\nTraining process completed successfully!")