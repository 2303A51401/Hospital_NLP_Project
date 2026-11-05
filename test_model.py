import joblib
import re
import nltk

from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")

# Load model
model = joblib.load(
    "model/hospital_intent_model.pkl"
)

# Load TF-IDF vectorizer
vectorizer = joblib.load(
    "model/tfidf_vectorizer.pkl"
)

# Stopwords
stop_words = set(
    stopwords.words("english")
)


# Text preprocessing
def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Ask patient query
query = input(
    "\nEnter patient query: "
)

# Clean query
cleaned_query = clean_text(query)

# Convert query into TF-IDF
query_tfidf = vectorizer.transform(
    [cleaned_query]
)

# Predict intent
prediction = model.predict(
    query_tfidf
)

print("\n================================")
print("NLP PREDICTION")
print("================================")

print(
    "\nPatient Query:"
)

print(query)

print(
    "\nCleaned Query:"
)

print(cleaned_query)

print(
    "\nPredicted Intent:"
)

print(prediction[0])

print(
    "\n================================"
)