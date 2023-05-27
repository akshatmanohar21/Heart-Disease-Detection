import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_file = 'depl.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Define the prediction function
def predict(data):
    # Preprocess the input data if necessary
    data = np.asarray(data)  # Convert data to NumPy array
    data = data.reshape(1, -1)  # Reshape the data to match the expected input shape
    data = data.astype(float)  # Convert data to float type

    # Make predictions using the trained model
    prediction = model.predict(data)
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
    age = st.slider("Age", min_value=0, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chol = st.number_input("Cholesterol", min_value=0, step=1)
    cp = st.slider("Chest Pain", min_value=0, step=1)

    # Convert user inputs to a feature vector
    # Example: Convert age, gender, cholesterol to feature vector
    feature_vector = [age, sex, chol, cp]

    # Make predictions on the feature vector
    prediction = predict(feature_vector)

    # Display the prediction
    st.write("Prediction:", prediction)

# Run the app
if __name__ == "__main__":
    main()
