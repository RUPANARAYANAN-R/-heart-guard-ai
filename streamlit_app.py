#  streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Guard AI ❤️",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model and Scaler ---
@st.cache_resource
def load_model():
    """Load the pre-trained model, scaler, and imputer."""
    try:
        model = joblib.load('heart_risk_model.joblib')
        scaler = joblib.load('scaler.joblib')
        imputer = joblib.load('imputer.joblib')
        return model, scaler, imputer
    except FileNotFoundError:
        st.error("Model files not found! Please run `model_training.py` first.")
        return None, None, None

model, scaler, imputer = load_model()
if model is None:
    st.stop()
    
# Load the training data to show in the app
try:
    df_train = pd.read_csv('synthetic_heart_data.csv')
except FileNotFoundError:
    st.warning("Training data not found. Visualizations might be limited.")
    df_train = None

# --- UI Layout ---
st.title("❤️ Heart Guard AI")
st.markdown("Your personal AI-powered heart health companion. Enter your details to get an instant risk assessment.")
st.markdown("---")

# --- Sidebar for User Input ---
st.sidebar.header("📋 Your Health Profile")

def user_input_features():
    """Create sidebar inputs and return a dictionary of values."""
    age = st.sidebar.slider('Age', 25, 80, 50)
    sex = st.sidebar.radio('Sex', ('Female', 'Male'))
    cholesterol = st.sidebar.slider('Cholesterol (mg/dL)', 150, 400, 200)
    systolic_bp = st.sidebar.slider('Systolic Blood Pressure (mmHg)', 90, 200, 120)
    diastolic_bp = st.sidebar.slider('Diastolic Blood Pressure (mmHg)', 60, 120, 80)
    heart_rate = st.sidebar.slider('Heart Rate (bpm)', 50, 120, 75)
    bmi = st.sidebar.slider('BMI', 18.0, 40.0, 25.0, 0.1)
    smoking = st.sidebar.radio('Do you smoke?', ('No', 'Yes'))
    family_history = st.sidebar.radio('Family History of Heart Disease?', ('No', 'Yes'))

    data = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'cholesterol': cholesterol,
        'blood_pressure_systolic': systolic_bp,
        'blood_pressure_diastolic': diastolic_bp,
        'heart_rate': heart_rate,
        'bmi': bmi,
        'smoking': 1 if smoking == 'Yes' else 0,
        'family_history': 1 if family_history == 'Yes' else 0
    }
    return data

input_data = user_input_features()
input_df = pd.DataFrame([input_data])

# --- Prediction and Output ---
st.header("📈 Your Risk Assessment")

if st.sidebar.button('Get Prediction', use_container_width=True):
    # Prepare input for prediction
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    risk_levels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    risk_colors = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
    risk_icons = {'Low Risk': '✅', 'Medium Risk': '⚠️', 'High Risk': '🚨'}
    
    predicted_risk = risk_levels[prediction]
    risk_color = risk_colors[predicted_risk]
    risk_icon = risk_icons[predicted_risk]
    
    # --- Display Prediction ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Gauge Chart for Risk Score
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Level", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 2], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickvals': [0, 1, 2], 'ticktext': ['Low', 'Med', 'High']},
                'bar': {'color': risk_color},
                'steps' : [
                    {'range': [-0.5, 0.5], 'color': 'lightgreen'},
                    {'range': [0.5, 1.5], 'color': 'lightgoldenrodyellow'},
                    {'range': [1.5, 2.5], 'color': 'lightcoral'}
                ],
            }
        ))
        gauge_fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col2:
        st.markdown(f"### {risk_icon} Result: <span style='color:{risk_color};'>{predicted_risk}</span>", unsafe_allow_html=True)
        st.markdown(f"Our AI model predicts that you have a **{predicted_risk.lower()}** of developing a heart condition based on the provided data.")
        st.progress(prediction / 2)

        st.subheader("💡 Recommendations")
        if prediction == 0:
            st.success("Great job! Continue to maintain a healthy lifestyle. Regular check-ups are still recommended.")
        elif prediction == 1:
            st.warning("Your risk is moderate. Consider lifestyle changes like improving your diet, increasing physical activity, and regular monitoring. Consult a doctor for personalized advice.")
        else:
            st.error("Your risk is high. It is strongly recommended to consult a healthcare professional for a comprehensive evaluation and personalized health plan.")

    st.markdown("---")
    
    # --- Visualizations ---
    st.header("📊 Deeper Insights")
    
    tab1, tab2, tab3 = st.tabs(["Your Profile vs. Population", "Risk Factor Importance", "Explore Training Data"])

    with tab1:
        st.subheader("How Your Vitals Compare")
        # Radar Chart
        categories = ['Cholesterol', 'Systolic BP', 'Diastolic BP', 'Heart Rate', 'BMI']
        user_values = [
            input_data['cholesterol'],
            input_data['blood_pressure_systolic'],
            input_data['blood_pressure_diastolic'],
            input_data['heart_rate'],
            input_data['bmi']
        ]

        # Define ideal ranges for the radar chart
        ideal_ranges = {
            'Cholesterol': 200, 'Systolic BP': 120, 'Diastolic BP': 80, 
            'Heart Rate': 70, 'BMI': 22
        }

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_values,
            theta=categories,
            fill='toself',
            name='Your Values'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=list(ideal_ranges.values()),
            theta=categories,
            fill='toself',
            name='Ideal Values'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(user_values)*1.2])),
            showlegend=True,
            title="Your Vitals vs. Ideal Range"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.subheader("What Factors Matter Most?")
        # Feature Importance Plot
        if hasattr(model, 'feature_importances_'):
            feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, input_df.columns)), columns=['Value','Feature'])
            fig_imp = px.bar(feature_imp, x="Value", y="Feature", orientation='h', title="Model's Risk Factor Importance")
            fig_imp.update_layout(xaxis_title="Importance", yaxis_title="Factor")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance is not available for this model type.")

    with tab3:
        st.subheader("Dataset Used for Training")
        if df_train is not None:
            st.markdown("This is the synthetic dataset used to train the Heart Guard AI model. You can explore how different factors correlate with risk.")
            st.dataframe(df_train)
        else:
            st.warning("Training data could not be loaded.")
else:
    st.info("Please fill in your details in the sidebar and click 'Get Prediction' to see your heart health assessment.")
    st.image("https://images.unsplash.com/photo-1549488344-cbb6c144eda4?q=80&w=2070", caption="Stay proactive about your heart health.")