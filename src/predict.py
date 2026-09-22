import joblib

model = joblib.load("model/ticket_classifier.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


def predict_ticket(description):
    description = description.lower()
    features = vectorizer.transform([description])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities) * 100
    return prediction, confidence


if __name__ == "__main__":
    print("=" * 60)
    print("       CUSTOMER SUPPORT TICKET CLASSIFIER")
    print("=" * 60)

    print("\nType 'exit' to close the program.")

    while True:
        description = input("\nEnter Ticket Description: ")

        if description.lower() == "exit":
            print("\nProgram closed.")
            break

        if not description.strip():
            print("Please enter a ticket description.")
            continue

        category, confidence = predict_ticket(description)

        print("\n------------------------------")
        print("Predicted Category:", category)
        print(f"Confidence: {confidence:.2f}%")
        print("------------------------------")