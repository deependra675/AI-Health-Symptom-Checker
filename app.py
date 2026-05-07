import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import time

# LOAD MODELS

model = joblib.load("model.pkl")
symptoms = joblib.load("symptoms.pkl")
description_dict = joblib.load("description.pkl")
precaution_dict = joblib.load("precaution.pkl")

# PAGE CONFIG

st.set_page_config(
    page_title="AI Health Assistant", 
    page_icon="🩺",
    layout="wide"
)
st.markdown("""
### AI-powered disease prediction using Machine Learning

Developed by **Deependra Sisodia (MCA)**
""")

# MODERN LIGHT UI STYLE

st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #f4f7fb;
    color: #1f2d3d;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 15px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    transition: 0.3s ease;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
}

/* Titles */
.title {
    font-size: 20px;
    font-weight: 600;
    color: #2c3e50;
}

/* Buttons */
.stButton>button {
    background-color: #2e86de;
    color: white;
    border-radius: 10px;
    padding: 8px 16px;
    border: none;
    font-weight: 500;
}

.stButton>button:hover {
    background-color: #1b4f72;
}

</style>
""", unsafe_allow_html=True)

# SIDEBAR INPUT

st.sidebar.title("🩺 Health Input Panel")

selected_symptoms = st.sidebar.multiselect(
    "Select Symptoms",
    symptoms
)

predict_btn = st.sidebar.button("Analyze Health")

# MAIN TITLE

st.title("🧠 AI Health Symptom Checker")
st.write("Select symptoms and get instant AI-powered analysis.")

# PREDICTION LOGIC

if predict_btn:

    if not selected_symptoms:
        st.warning("Please select at least one symptom.")

    else:
        
        # LOADING ANIMATION
    
        with st.spinner("🧠 AI is analyzing your symptoms..."):
            time.sleep(1.5)

            # INPUT VECTOR
            input_data = [0] * len(symptoms)

            for sym in selected_symptoms:
                if sym in symptoms:
                    idx = symptoms.index(sym)
                    input_data[idx] = 1

            # PREDICTION
            prediction = model.predict([input_data])[0]
            probabilities = model.predict_proba([input_data])[0]
            confidence = max(probabilities) * 100
        
        # TABS UI
     
        tab1, tab2, tab3 = st.tabs([
            "🧠 Diagnosis",
            "📖 Medical Info",
            "🛡️ Precautions"
        ])
        # TAB 1 - DIAGNOSIS
        
        with tab1:

            st.markdown(f"""
            <div class="card">
                <div class="title">Predicted Condition</div>
                <h2 style="color:#2e86de">{prediction}</h2>
                <p><b>Confidence:</b> {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Confidence Score", f"{confidence:.2f}%")

        # TAB 2 - DESCRIPTION
     
        with tab2:

            description = description_dict.get(
                prediction,
                "No description available."
            )

            st.markdown(f"""
            <div class="card">
                <div class="title">Disease Overview</div>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

        # TAB 3 - PRECAUTIONS
        
        with tab3:

            precautions = precaution_dict.get(prediction, [])

            if precautions:

                st.markdown("""
                <div class="card">
                    <div class="title">Recommended Precautions</div>
                </div>
                """, unsafe_allow_html=True)

                for p in precautions:
                    st.write("✔", p)

            else:
                st.warning("No precautions available.")

        # CONFIDENCE CHART
        
        st.subheader("📊 AI Prediction Analysis")

        top_indices = probabilities.argsort()[-5:][::-1]
        top_diseases = model.classes_[top_indices]
        top_scores = probabilities[top_indices] * 100

        chart_df = pd.DataFrame({
            "Disease": top_diseases,
            "Confidence": top_scores
        })

        fig = px.bar(
            chart_df,
            x="Disease",
            y="Confidence",
            color="Disease",
            text="Confidence"
        )

        fig.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font_color="#1f2d3d"
        )

        st.plotly_chart(fig, use_container_width=True)


# DISCLAIMER

st.markdown("""
---
⚠️ This tool is for educational purposes only and not a substitute for professional medical advice.
""")
