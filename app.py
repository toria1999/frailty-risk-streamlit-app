import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


APP_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Older Adult Frailty Risk Explorer",
    page_icon="🧓",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load(APP_DIR / "frailty_model.pkl")
    feature_columns = joblib.load(APP_DIR / "feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

st.title("Older Adult Frailty Risk Explorer")
st.write(
    "This Streamlit app uses a machine-learning model to estimate the probability "
    "of robust, pre-frail, and frail status among older adults based on demographic, "
    "chronic disease, physical activity, and physical function inputs."
)

st.sidebar.header("Enter older adult profile")

age = st.sidebar.slider("Age", 60, 90, 72)
bmi = st.sidebar.number_input("BMI", min_value=12.0, max_value=60.0, value=27.5, step=0.1)

sex = st.sidebar.selectbox("Sex", ["Female", "Male"])

race_ethnicity = st.sidebar.selectbox(
    "Race/ethnicity",
    [
        "Mexican American",
        "Other Hispanic",
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Non-Hispanic Asian",
        "Other race or multiracial",
    ],
)

education = st.sidebar.selectbox(
    "Education",
    [
        "Less than 9th grade",
        "9th to 11th grade",
        "High school or GED",
        "Some college or AA degree",
        "College graduate or above",
    ],
)

diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
stroke = st.sidebar.selectbox("Stroke history", ["No", "Yes"])
physically_active = st.sidebar.selectbox("Physically active", ["Yes", "No"])

st.sidebar.subheader("Physical function difficulties")

diff_walk_quarter_mile = st.sidebar.selectbox("Difficulty walking a quarter mile", ["No", "Yes"])
diff_walk_10_steps = st.sidebar.selectbox("Difficulty walking up 10 steps", ["No", "Yes"])
diff_stoop = st.sidebar.selectbox("Difficulty stooping/crouching/kneeling", ["No", "Yes"])
diff_lift_carry = st.sidebar.selectbox("Difficulty lifting or carrying", ["No", "Yes"])
diff_household_chores = st.sidebar.selectbox("Difficulty with household chores", ["No", "Yes"])

input_df = pd.DataFrame(
    {
        "age": [age],
        "sex": [sex],
        "race_ethnicity": [race_ethnicity],
        "education": [education],
        "bmi": [bmi],
        "diabetes": [diabetes],
        "hypertension": [hypertension],
        "stroke": [stroke],
        "physically_active": [physically_active],
        "diff_walk_quarter_mile": [diff_walk_quarter_mile],
        "diff_walk_10_steps": [diff_walk_10_steps],
        "diff_stoop": [diff_stoop],
        "diff_lift_carry": [diff_lift_carry],
        "diff_household_chores": [diff_household_chores],
    }
)

input_encoded = pd.get_dummies(input_df, drop_first=False)

for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[feature_columns]

predicted_class = model.predict(input_encoded)[0]
predicted_probs = model.predict_proba(input_encoded)[0]
class_labels = model.classes_

prob_df = pd.DataFrame(
    {
        "Frailty category": class_labels,
        "Predicted probability": predicted_probs,
    }
).sort_values("Predicted probability", ascending=False)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Predicted frailty category")
    st.metric(label="Model prediction", value=predicted_class)

    st.subheader("Predicted probabilities")
    st.dataframe(prob_df, use_container_width=True)

with col2:
    st.subheader("Probability chart")
    fig, ax = plt.subplots()
    ax.bar(prob_df["Frailty category"], prob_df["Predicted probability"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted probability")
    ax.set_xlabel("Frailty category")
    st.pyplot(fig)

st.subheader("Interpretation")

if predicted_class == "Robust":
    st.write(
        "The model predicts the highest probability for the robust category. "
        "This profile shows relatively low frailty-related functional difficulty."
    )
elif predicted_class == "Pre-frail":
    st.write(
        "The model predicts the highest probability for the pre-frail category. "
        "This profile may represent an intermediate level of frailty-related risk."
    )
else:
    st.write(
        "The model predicts the highest probability for the frail category. "
        "This profile shows multiple frailty-related risk indicators."
    )

st.warning(
    "This app is an educational machine-learning demonstration. "
    "It is not a clinical diagnostic tool and should not be used for medical decision-making."
)

with st.expander("About this app"):
    st.write(
        """
        This app demonstrates how a machine-learning model can be used to classify older adult
        frailty status based on demographic, chronic disease, physical activity, and physical
        function variables.

        The model is intended for a class capstone project. It does not provide clinical diagnosis.
        If survey design variables are not incorporated, the predictions should not be interpreted
        as nationally representative NHANES prevalence estimates.
        """
    )

with st.expander("Model information"):
    st.write(
        """
        Model type: Random Forest classifier

        Outcome: Robust, pre-frail, or frail status

        Predictors: age, sex, race/ethnicity, education, BMI, diabetes, hypertension,
        stroke history, physical activity, and physical function difficulties.
        """
    )
