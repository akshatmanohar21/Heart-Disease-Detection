import streamlit as st
import pickle

# Load the trained model
model_file = 'depl.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Define the prediction function
def predict(data):
    # Preprocess the input data if necessary
    # Make predictions using the trained model
    prediction = model.predict([data])
    return prediction

# Create the Streamlit web app
def main():
    # Set the app title
    st.title("Heart Disease Prediction App")

    # Add description and instructions
    st.write("This app predicts whether a person has heart disease or not.")
    st.write("Please enter the required information.")

    # Add input fields for user input
    # Example: age, gender, cholesterol, etc.
    age = st.slider("Age", min_value=0, max_value=100, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chol = st.number_input("Cholesterol", min_value=0, step=1)
    cp = st.slider("Chest Pain", min_value=0, max_value=3, step=1)
    # Add more input fields as needed

    # Convert user inputs to a feature vector
    feature_vector = [age, sex, chol, cp]
    # Convert the feature vector to the desired input format for the model

    # Make predictions on the feature vector
    prediction = predict(feature_vector)

    # Display the prediction
    if prediction:
        st.write("Prediction: Person has heart disease.")
    else:
        st.write("Prediction: Person does not have heart disease.")

# Run the app
if __name__ == "__main__":
    main()
