from flask import Flask, render_template, request
import joblib
import re
import nltk

from nltk.corpus import stopwords


# ==========================================
# CREATE FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "model/hospital_intent_model.pkl"
)

vectorizer = joblib.load(
    "model/tfidf_vectorizer.pkl"
)


# ==========================================
# DOWNLOAD STOPWORDS
# ==========================================

nltk.download("stopwords")

stop_words = set(
    stopwords.words("english")
)


# ==========================================
# TEXT PREPROCESSING FUNCTION
# ==========================================

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

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Join words
    return " ".join(words)


# ==========================================
# HOSPITAL INFORMATION
# FOR ALL 10 INTENTS
# ==========================================

responses = {

    "appointment_booking":
        "Appointment Booking:\n"
        "You can book an appointment by providing your "
        "preferred doctor, department, date and time.\n"
        "Please carry your patient ID during the visit.",


    "doctor_appointment":
        "Doctor Appointment:\n"
        "Doctor appointments are available for different "
        "departments and specialties.\n"
        "Please select your preferred doctor and appointment date.",


    "doctor_availability":
        "Doctor Availability:\n"
        "Please provide the doctor name or department "
        "to check the available appointment slots.\n"
        "Availability depends on the doctor's schedule.",


    "hospital_timings":
        "Hospital Timings:\n"
        "Monday - Saturday: 9:00 AM - 8:00 PM\n"
        "Sunday: 10:00 AM - 2:00 PM\n"
        "Emergency services are available 24 hours.",


    "hospital_location":
        "Hospital Location:\n"
        "Main Road, City Center\n"
        "Near Central Bus Station\n"
        "City, India\n"
        "Contact: 1800-123-4567",


    "contact_information":
        "Contact Information:\n"
        "Hospital Contact Number: 1800-123-4567\n"
        "Email: info@hospital.com\n"
        "Reception Desk: 9:00 AM - 8:00 PM",


    "insurance":
        "Insurance Information:\n"
        "The hospital accepts major health insurance plans.\n"
        "Please contact the insurance desk to confirm "
        "your insurance coverage before treatment.",


    "payment":
        "Payment Information:\n"
        "Payment can be made using Cash, Credit Card, "
        "Debit Card, UPI or other available payment methods.\n"
        "Please contact the billing department for details.",


    "appointment_cancellation":
        "Appointment Cancellation:\n"
        "You can cancel your appointment by providing "
        "your appointment ID.\n"
        "Please cancel the appointment in advance whenever possible.",


    "appointment_rescheduling":
        "Appointment Rescheduling:\n"
        "You can reschedule your appointment by providing "
        "your appointment ID and preferred new date and time.\n"
        "The new slot depends on doctor availability."
}


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICT PATIENT QUERY
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Get query from webpage
    query = request.form.get(
        "query",
        ""
    )

    # Check empty query
    if not query.strip():

        return render_template(
            "index.html",
            error="Please enter a patient query."
        )


    # ==========================================
    # CLEAN QUERY
    # ==========================================

    cleaned_query = clean_text(
        query
    )


    # ==========================================
    # CONVERT QUERY TO TF-IDF
    # ==========================================

    query_tfidf = vectorizer.transform(
        [cleaned_query]
    )


    # ==========================================
    # PREDICT INTENT
    # ==========================================

    prediction = model.predict(
        query_tfidf
    )

    predicted_intent = prediction[0]


    # ==========================================
    # GET RESPONSE
    # ==========================================

    response = responses.get(
        predicted_intent,
        "Please contact the hospital for more information."
    )


    # ==========================================
    # DISPLAY RESULT
    # ==========================================

    return render_template(
        "index.html",
        query=query,
        intent=predicted_intent,
        response=response
    )


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )