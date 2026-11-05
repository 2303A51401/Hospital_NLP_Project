# ==========================================
# HOSPITAL APPOINTMENT AND PATIENT QUERY
# CLASSIFICATION USING NLP
# ==========================================


# ==========================================
# STEP 1: IMPORT LIBRARIES
# ==========================================

import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# STEP 2: DOWNLOAD STOPWORDS
# ==========================================

nltk.download("stopwords")


# ==========================================
# STEP 3: LOAD DATASET
# ==========================================

df = pd.read_csv(
    "dataset/hospital_dataset_updated.csv"
)


# ==========================================
# STEP 4: DISPLAY DATASET INFORMATION
# ==========================================

print("======================================")
print("DATASET INFORMATION")
print("======================================")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nIntent Distribution:")
print(df["Intent"].value_counts())


# ==========================================
# STEP 5: TEXT PREPROCESSING
# ==========================================

# Get English stopwords
stop_words = set(
    stopwords.words("english")
)


# Text preprocessing function
def clean_text(text):

    # Convert text to lowercase
    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Split sentence into words
    words = text.split()

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Join words back together
    return " ".join(words)


# Apply preprocessing
df["Clean_Query"] = df[
    "Raw_Patient_Query"
].apply(clean_text)


# Display original and cleaned queries
print("\n======================================")
print("ORIGINAL AND CLEANED QUERIES")
print("======================================")

print(
    df[
        [
            "Raw_Patient_Query",
            "Clean_Query"
        ]
    ].head(10)
)


# ==========================================
# STEP 6: INPUT AND TARGET
# ==========================================

# Input
X = df["Clean_Query"]

# Target
y = df["Intent"]


# ==========================================
# STEP 7: TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n======================================")
print("TRAIN-TEST SPLIT")
print("======================================")

print("\nTraining records:")
print(len(X_train))

print("\nTesting records:")
print(len(X_test))


# ==========================================
# STEP 8: TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(

    ngram_range=(1, 2),

    max_features=5000
)


# Fit and transform training data
X_train_tfidf = vectorizer.fit_transform(
    X_train
)


# Transform testing data
X_test_tfidf = vectorizer.transform(
    X_test
)


print("\n======================================")
print("TF-IDF VECTORIZATION")
print("======================================")

print("\nTF-IDF Training Shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF Testing Shape:")
print(X_test_tfidf.shape)


# ==========================================
# STEP 9: LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(

    max_iter=1000
)


# Train model
model.fit(
    X_train_tfidf,
    y_train
)


print("\n======================================")
print("MODEL TRAINING")
print("======================================")

print(
    "\nLogistic Regression model "
    "training completed successfully!"
)


# ==========================================
# STEP 10: PREDICTION
# ==========================================

y_pred = model.predict(
    X_test_tfidf
)


print("\nPrediction completed successfully!")


# ==========================================
# STEP 11: MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")


# Accuracy
print("\nAccuracy:")
print(accuracy)


print("\nAccuracy Percentage:")
print(
    f"{accuracy * 100:.2f}%"
)


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# Confusion matrix
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# STEP 12: SAVE TRAINED MODEL
# ==========================================

joblib.dump(

    model,

    "model/hospital_intent_model.pkl"
)


# Save TF-IDF vectorizer
joblib.dump(

    vectorizer,

    "model/tfidf_vectorizer.pkl"
)


print("\n======================================")
print("MODEL SAVED SUCCESSFULLY")
print("======================================")


print(
    "\nModel saved at:"
)

print(
    "model/hospital_intent_model.pkl"
)


print(
    "\nVectorizer saved at:"
)

print(
    "model/tfidf_vectorizer.pkl"
)


# ==========================================
# PROJECT COMPLETION MESSAGE
# ==========================================

print("\n======================================")
print("NLP TRAINING COMPLETED")
print("======================================")

print(
    "\nYour hospital intent classification "
    "model is ready!"
)