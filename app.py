# Import required libraries
import streamlit as st
import numpy as np
import pickle

# Set page configuration (this must be the first Streamlit command)
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide"
)

# Load the trained model


@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(
            "❌ model.pkl not found! Please make sure the model file is in the same folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


# Load the model
model = load_model()
st.success("✅ Model loaded successfully!")

# App title and description
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This app predicts whether a patient has **heart disease** based on medical information.
Enter the patient's details below and click **Predict** to see the result.
""")
st.markdown("---")

# Create input sections
st.subheader("📋 Patient Information")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Basic Information")

    age = st.number_input(
        "Age (years)",
        min_value=0,
        max_value=120,
        value=50,
        step=1,
        help="Age of the patient"
    )

    sex = st.selectbox(
        "Sex",
        options=["Female", "Male"],
        help="Gender of the patient"
    )
    # Convert to 0 or 1 for model
    sex_value = 1 if sex == "Male" else 0

    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 - Typical angina",
            1: "1 - Atypical angina",
            2: "2 - Non-anginal pain",
            3: "3 - Asymptomatic"
        }[x],
        help="Type of chest pain experienced"
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (trestbps) - mm Hg",
        min_value=0,
        max_value=300,
        value=120,
        step=1,
        help="Resting blood pressure (in mm Hg on admission)"
    )

    chol = st.number_input(
        "Serum Cholesterol (chol) - mg/dl",
        min_value=0,
        max_value=600,
        value=200,
        step=1,
        help="Serum cholesterol in mg/dl"
    )

with col2:
    st.markdown("### Medical Test Results")

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl (fbs)",
        options=["No", "Yes"],
        help="Fasting blood sugar level"
    )
    fbs_value = 1 if fbs == "Yes" else 0

    restecg = st.selectbox(
        "Resting ECG Results (restecg)",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 - Normal",
            1: "1 - ST-T wave abnormality",
            2: "2 - Left ventricular hypertrophy"
        }[x],
        help="Resting electrocardiographic results"
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved (thalach)",
        min_value=0,
        max_value=250,
        value=150,
        step=1,
        help="Maximum heart rate achieved"
    )

    exang = st.selectbox(
        "Exercise Induced Angina (exang)",
        options=["No", "Yes"],
        help="Angina induced by exercise"
    )
    exang_value = 1 if exang == "Yes" else 0

    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="ST depression induced by exercise relative to rest"
    )

# Second row of inputs
st.markdown("---")
st.subheader("Additional Clinical Measurements")

col3, col4 = st.columns(2)

with col3:
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment (slope)",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 - Upsloping",
            1: "1 - Flat",
            2: "2 - Downsloping"
        }[x],
        help="The slope of the peak exercise ST segment"
    )

    ca = st.number_input(
        "Number of Major Vessels (ca)",
        min_value=0,
        max_value=4,
        value=0,
        step=1,
        help="Number of major vessels colored by fluoroscopy (0-4)"
    )

with col4:
    thal = st.selectbox(
        "Thalassemia (thal)",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 - Normal",
            2: "2 - Fixed defect",
            3: "3 - Reversible defect"
        }[x],
        help="Thalassemia type"
    )

# Prediction button
st.markdown("---")
predict_button = st.button("🔍 Predict Heart Disease Risk", type="primary")

# Make prediction when button is clicked
if predict_button:
    # Create feature array in the exact order your model expects
    features = np.array([[
        age,           # age
        sex_value,     # sex
        cp,            # cp
        trestbps,      # trestbps
        chol,          # chol
        fbs_value,     # fbs
        restecg,       # restecg
        thalach,       # thalach
        exang_value,   # exang
        oldpeak,       # oldpeak
        slope,         # slope
        ca,            # ca
        thal           # thal
    ]])

    # Show a spinner while predicting
    with st.spinner("Analyzing patient data..."):
        try:
            # Make prediction
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features)

            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            # Create columns for results
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if prediction[0] == 1:
                    st.error("⚠️ **HIGH RISK**")
                    st.markdown(
                        "The model predicts that the patient **has heart disease**.")
                else:
                    st.success("✅ **LOW RISK**")
                    st.markdown(
                        "The model predicts that the patient **does not have heart disease**.")

            with result_col2:
                st.markdown("### Confidence Levels")
                st.write(f"**No Heart Disease:** {prediction_proba[0][0]:.2%}")
                st.write(f"**Heart Disease:** {prediction_proba[0][1]:.2%}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Make sure the model was trained with the same features.")

# Add sidebar with information
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown("""
    ### How to Use
    1. Enter all patient information
    2. Click the **Predict** button
    3. View the prediction result
    
    ### Model Information
    - **Task:** Binary Classification
    - **Output:** Heart Disease (Yes/No)
    - **Features:** 13 clinical parameters
    
    ### Features Explained
    - **Age:** Age in years
    - **Sex:** Male/Female
    - **CP:** Chest pain type
    - **Trestbps:** Resting blood pressure
    - **Chol:** Serum cholesterol
    - **FBS:** Fasting blood sugar
    - **Restecg:** Resting ECG results
    - **Thalach:** Max heart rate achieved
    - **Exang:** Exercise induced angina
    - **Oldpeak:** ST depression
    - **Slope:** ST segment slope
    - **CA:** Major vessels count
    - **Thal:** Thalassemia type
    
    ### Important Note
    This is a predictive tool and should not replace professional medical diagnosis.
    """)

    # Show model info if available
    if hasattr(model, 'n_features_in_'):
        st.write(f"**Model expects:** {model.n_features_in_} features")
