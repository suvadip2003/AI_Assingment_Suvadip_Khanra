import streamlit as st
import joblib




st.set_page_config(
    page_title="AI Customer Support Classifier",
    page_icon="🤖",
    layout="centered"
)




@st.cache_resource
def load_model():

    model = joblib.load(
        "model/ticket_classifier.pkl"
    )

    vectorizer = joblib.load(
        "model/tfidf_vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_model()


st.title(
    "🤖 AI Customer Support Ticket Classifier"
)

st.write(
    "Enter a customer support ticket description "
    "to automatically predict its category."
)


ticket_description = st.text_area(
    "Enter Ticket Description",
    placeholder=(
        "Example: I forgot my password and "
        "cannot access my account."
    ),
    height=150
)



if st.button(
    "Predict Ticket",
    type="primary"
):

    if not ticket_description.strip():

        st.warning(
            "Please enter a ticket description."
        )

    else:

       
        features = vectorizer.transform(
            [ticket_description.lower()]
        )

       
        prediction = model.predict(
            features
        )[0]

        
        probabilities = model.predict_proba(
            features
        )[0]

        confidence = max(
            probabilities
        ) * 100


        
        st.success(
            "Prediction completed successfully!"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Category",
                prediction
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


     

        if confidence >= 80:

            st.info(
                "The model is highly confident "
                "about this prediction."
            )

        elif confidence >= 60:

            st.info(
                "The model has moderate confidence "
                "about this prediction."
            )

        else:

            st.warning(
                "The model has relatively low confidence. "
                "This ticket may require manual review."
            )



st.divider()

st.subheader(
    "Supported Categories"
)

st.write(
    "• Login Issue\n"
    "• Application Error\n"
    "• Report\n"
    "• Account Update\n"
    "• Performance"
)

st.caption(
    "Machine Learning Model: TF-IDF + Logistic Regression"
)