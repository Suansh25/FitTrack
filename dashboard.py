import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap


st.set_page_config(
    page_title="FitTrack",

    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.big-font {
    font-size: 22px !important;
    font-weight: 600;
}
.section-card {
    background-color: #161b22;
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    try:
        m = joblib.load('calories_model.pkl')
        e = joblib.load('shap_explainer.pkl')
        f = joblib.load('feature_names.pkl')
        return m, e, f
    except:
        return None, None, None

@st.cache_data
def load_food_data():
    try:
        df = pd.read_csv('assets/Indian_Food_Nutrition_Processed.csv')
        df.columns = [c.strip() for c in df.columns]
        return df
    except:
        return pd.DataFrame()

model, explainer, feature_names = load_assets()
food_df = load_food_data()

if model is None:
    st.error("Model assets not found.")
    st.stop()


st.sidebar.title("Profile")

sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.number_input("Age", 10, 100, 25)
height = st.sidebar.number_input("Height (cm)", 100.0, 250.0, 175.0)
weight = st.sidebar.number_input("Weight (kg)", 30.0, 200.0, 75.0)

st.sidebar.markdown("---")
st.sidebar.title("Goal")

goal = st.sidebar.radio(
    "Select Goal",
    ["Maintenance", "Weight Loss", "Bulking"]
)


st.markdown("# FitTrack")
st.caption("## Predictive Modelling of Daily Calorie Requirements using User Physiological Data")


with st.container():
    st.markdown("### Activity Tracker")
    col1, col2, col3 = st.columns(3)

    duration = col1.slider("Duration (mins)", 0, 180, 45)
    heart_rate = col2.slider("Avg Heart Rate", 60, 200, 110)
    body_temp = col3.slider("Body Temp (°C)", 36.0, 42.0, 39.0)


sex_enc = 1 if sex == "male" else 0
bmi = weight / ((height/100) ** 2)
intensity_factor = duration * heart_rate

input_df = pd.DataFrame([[
    sex_enc, age, height, weight, bmi,
    duration, heart_rate, body_temp,
    intensity_factor
]], columns=feature_names)

exercise_burn = model.predict(input_df)[0]


def calculate_bmr(weight, height, age, sex):
    if sex == 'male':
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    return (10 * weight) + (6.25 * height) - (5 * age) - 161

bmr = calculate_bmr(weight, height, age, sex)
base_burn = bmr * 1.2

goal_adjustment = 0
if goal == "Weight Loss":
    goal_adjustment = -500
elif goal == "Bulking":
    goal_adjustment = 500

total_daily_target = base_burn + exercise_burn + goal_adjustment


st.markdown("###  Daily Metrics")
m1, m2, m3 = st.columns(3)

m1.metric("Resting Metabolism (BMR)", f"{int(bmr)} kcal")
m2.metric("Exercise Burn (AI)", f"{int(exercise_burn)} kcal")
m3.metric("Daily Target", f"{int(total_daily_target)} kcal")


st.markdown("### Prediction Insights")

with st.container():

    shap_values = explainer.shap_values(input_df)
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values[0]
    })

    shap_df["AbsImpact"] = shap_df["Impact"].abs()
    shap_df = shap_df.sort_values("AbsImpact", ascending=False)

   
    top_features = shap_df.head(6)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Key Factors Influencing Burn")

        
        chart_df = top_features.set_index("Feature")[["Impact"]]
        st.bar_chart(chart_df)

    with col2:
        st.markdown("####  Interpretation")

        positive = top_features[top_features["Impact"] > 0]
        negative = top_features[top_features["Impact"] < 0]

        if not positive.empty:
            st.success(
                f"⬆ Increasing Burn: {', '.join(positive['Feature'].head(3))}"
            )

        if not negative.empty:
            st.error(
                f"⬇ Reducing Burn: {', '.join(negative['Feature'].head(3))}"
            )

        st.info(
            "Impact values show how strongly each factor influenced today's calorie burn prediction."
        )

st.markdown("###  Food Logger")

if 'food_log' not in st.session_state:
    st.session_state.food_log = []

col1, col2 = st.columns([2,1])

with col1:
    if not food_df.empty:
        selected = st.selectbox("Select Food", food_df['Dish'].unique())
        qty = st.number_input("Servings", 0.5, 10.0, 1.0, step=0.5)

        if st.button(" Add Food"):
            item = food_df[food_df['Dish'] == selected].iloc[0]
            cal_col = 'Calories ' if 'Calories ' in food_df.columns else 'Calories'

            st.session_state.food_log.append({
                "Food": selected,
                "Calories": item[cal_col] * qty,
                "Protein": item['Protein (g)'] * qty,
                "Carbs": item['Carbs (g)'] * qty,
                "Fat": item['Fat (g)'] * qty
            })
            st.success(f"Added {selected}")

with col2:
    if st.session_state.food_log:
        log_df = pd.DataFrame(st.session_state.food_log)
        total_eaten = log_df['Calories'].sum()
        total_protein = log_df['Protein'].sum()
        total_carbs = log_df['Carbs'].sum()
        total_fat = log_df['Fat'].sum()

        st.dataframe(log_df[['Food', 'Calories']], height=200)

        if st.button("Reset Log"):
            st.session_state.food_log = []
            st.rerun()
    else:
        total_eaten = total_protein = total_carbs = total_fat = 0
        st.info("No meals logged yet")


st.markdown("### Daily Progress")

remaining = total_daily_target - total_eaten
progress = min(total_eaten / total_daily_target, 1.0)

st.progress(progress)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Calories Eaten", f"{int(total_eaten)} kcal")
c2.metric("Remaining", f"{int(remaining)} kcal")
c3.metric("Protein", f"{int(total_protein)} g")
c4.metric("Carbs", f"{int(total_carbs)} g")


if remaining < 0:
    st.error(f" Over target by {abs(int(remaining))} kcal")
elif remaining < 200:
    st.warning(" Close to your calorie limit")
else:
    st.success(" You're on track!")


if total_eaten > 0:
    st.markdown("###  Macro Distribution")
    macro_df = pd.DataFrame({
        "Macro": ["Protein", "Carbs", "Fat"],
        "Grams": [total_protein, total_carbs, total_fat]
    })
    st.bar_chart(macro_df.set_index("Macro"))