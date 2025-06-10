import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# ----------------------- CONFIG ------------------------
DATA_DIR = "csvs_extracted/data"
MODELS = ["holtwinters"]  # You can extend this
REGIONS = ["rakhiyal", "bopal", "ambawadi", "chandkheda", "vastral"]
MONTH_MAP = {
    "January": "01", "February": "02", "March": "03", "April": "04",
    "May": "05", "June": "06", "July": "07", "August": "08",
    "September": "09", "October": "10", "November": "11", "December": "12"
}

# ----------------------- HELPER ------------------------
def load_csv(model, region, year):
    file = f"{model}_{region}_{year}.csv"
    path = os.path.join(DATA_DIR, file)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['date'])
        df['day'] = df['date'].dt.day
        return df
    return None

# ------------------- VIEW 1: 2024 PREDICTION -------------------
def view_2024():
    st.header("📅 2024 Forecast Viewer")

    model = st.selectbox("Select Model", MODELS)
    region = st.selectbox("Select Region", REGIONS)
    month_name = st.selectbox("Select Month", list(MONTH_MAP.keys()))
    month = MONTH_MAP[month_name]

    if st.button("🚀 GO", key="2024go"):
        df = load_csv(model, region, 2024)
        if df is None:
            st.error("CSV not found.")
            return

        df_month = df[df['date'].dt.month == int(month)]
        if df_month.empty:
            st.warning("No data for this month")
            return

        unique_days = sorted(df_month['day'].unique())
        day = st.selectbox("Select Day", unique_days)
        show_actual = st.checkbox("🔍 Show Actual Temperature (if available)")

        df_day = df_month[df_month['day'] == day]

        fig = px.line(df_day, x="hour", y="predicted_temperature", labels={"hour": "Hour", "predicted_temperature": "Predicted Temperature (°C)"}, title=f"{region.title()} | {month_name} {day}, 2024")

        if show_actual and 'actual_temperature' in df_day.columns:
            fig.add_scatter(x=df_day['hour'], y=df_day['actual_temperature'], name="Actual", mode='lines+markers')

        st.plotly_chart(fig)

        if st.button("🔄 Actual vs Predicted Graph"):
            st.session_state.view = "actual_vs_predicted"
            st.session_state.region = region
            st.session_state.month = month
            st.session_state.day = day

        if st.button("🔮 Show Prediction for Unseen Data (2025)"):
            st.session_state.view = "view_2025"
            st.session_state.region = region
            st.session_state.month = month
            st.session_state.day = day

# ------------------- VIEW 2: ACTUAL VS PREDICTED -------------------
def actual_vs_predicted():
    st.header("📊 Actual vs Predicted (2024)")

    region = st.session_state.region
    month = st.session_state.month
    day = st.session_state.day

    rmse_scores = []

    for model in MODELS:
        df = load_csv(model, region, 2024)
        df_month = df[df['date'].dt.month == int(month)]
        df_day = df_month[df_month['day'] == day]

        st.subheader(f"Model: {model.upper()}")
        fig = px.line(df_day, x="hour", y="predicted_temperature", labels={"hour": "Hour", "predicted_temperature": "Temperature (°C)"}, title=f"{region.title()} {day}/{month}/2024")

        if 'actual_temperature' in df_day.columns:
            fig.add_scatter(x=df_day['hour'], y=df_day['actual_temperature'], name="Actual", mode='lines+markers')
            rmse = np.sqrt(np.mean((df_day['actual_temperature'] - df_day['predicted_temperature'])**2))
            rmse_scores.append((model, rmse))

        st.plotly_chart(fig)

    if rmse_scores:
        st.markdown("### 📉 RMSE Scores")
        st.table(pd.DataFrame(rmse_scores, columns=["Model", "RMSE (°C)"]))

    if st.button("⬅ Back"):
        st.session_state.view = "view_2024"

# ------------------- VIEW 3: UNSEEN DATA PREDICTION (2025) -------------------
def view_2025():
    st.header("🔮 Unseen Data Prediction (2025)")

    model = st.selectbox("Select Model", MODELS)
    region = st.session_state.get("region", REGIONS[0])
    month = st.session_state.get("month", "01")

    df = load_csv(model, region, 2025)
    if df is None:
        st.error("CSV not found.")
        return

    df_month = df[df['date'].dt.month == int(month)]
    unique_days = sorted(df_month['day'].unique())
    day = st.selectbox("Select Day", unique_days)
    df_day = df_month[df_month['day'] == day]

    st.subheader(f"{model.upper()} Prediction: {region.title()} on {day}/{month}/2025")
    fig = px.line(df_day, x="hour", y="predicted_temperature", title="24-hour Forecast")
    st.plotly_chart(fig)

    if st.button("✅ Compare Models"):
        st.session_state.view = "compare_2025"
        st.session_state.region = region
        st.session_state.month = month
        st.session_state.day = day

# ------------------- VIEW 4: 2025 COMPARE MODELS -------------------
def compare_2025():
    st.header("📈 Compare All Models (2025 Prediction)")

    region = st.session_state.region
    month = st.session_state.month
    day = st.session_state.day

    for model in MODELS:
        df = load_csv(model, region, 2025)
        if df is not None:
            df_month = df[df['date'].dt.month == int(month)]
            df_day = df_month[df_month['day'] == day]

            st.subheader(f"Model: {model.upper()}")
            fig = px.line(df_day, x="hour", y="predicted_temperature", title=f"{region.title()} {day}/{month}/2025")
            st.plotly_chart(fig)

# ------------------- MAIN -------------------
def main():
    if 'view' not in st.session_state:
        st.session_state.view = "view_2024"

    view = st.session_state.view
    if view == "view_2024":
        view_2024()
    elif view == "actual_vs_predicted":
        actual_vs_predicted()
    elif view == "view_2025":
        view_2025()
    elif view == "compare_2025":
        compare_2025()

if __name__ == "__main__":
    main()
