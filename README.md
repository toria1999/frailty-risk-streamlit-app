# Older Adult Frailty Risk Explorer

## Project description

This project is a Streamlit machine-learning app that estimates frailty status among older adults. The app uses demographic, chronic disease, physical activity, and physical function inputs to predict whether an older adult profile is most consistent with robust, pre-frail, or frail status.

## What the app does

The user enters an older adult profile in the sidebar. A trained random forest classifier then returns the predicted frailty category and the predicted probabilities for robust, pre-frail, and frail status.

## Files included

- `app.py`: Streamlit application
- `data/cleaned_nhanes_frailty.csv`: cleaned analytic dataset
- `models/frailty_model.pkl`: trained random forest model
- `models/feature_columns.pkl`: model feature column list
- `requirements.txt`: required Python packages
- `README.md`: project description and instructions

## How to run the app

Install the required packages:

```bash
pip install -r requirements.txt