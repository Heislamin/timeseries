import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import glob

# --- CONFIGURATION ---
DATA_DIR = "csvs_extracted/data"

def detect_models(data_dir):
    files = glob.glob(f"{data_dir}/*_rakhiyal_2024.csv")
    return sorted(list(set([os.path.basename(f).split("_")[0] for f in files])))

MODELS = detect_models(DATA_DIR)
REGIONS = ["rakhiyal", "bopal", "ambawadi", "chandkheda", "vastral"]
MONTHS = {
    "January": "01", "February": "02", "March": "03", "April": "04",
    "May": "05", "June": "06", "July": "07", "August": "08",
    "September": "09", "October": "10", "November": "11", "December": "12"
}

st.set_page_config(page_title="Temperature Forecaster", layout="wide")
st.title("🌡️ Time Series Temperature Analyzer")

# --- MAIN MENU ---
view = st.sidebar.radio("Select View", ["📅 2024 Forecast", "📊 Actual vs Predicted (2024)", "🔮 2025 Unseen Forecast"])

# --- COMMON UI ---
region = st.sidebar.selectbox("Select Region", REGIONS)
month_name = st.sidebar.selectbox("Select Month", list(MONTHS.keys()))
month = MONTHS[month_name]

# ========================================
# 📅 VIEW 1: 2024 Forecast
# ========================================
if view == "📅 2024 Forecast":
    model = st.sidebar.selectbox("Select Model", MODELS)
    file_path = f"{DATA_DIR}/{model}_{region}_2024.csv"

    if not os.path.exists(file_path):
        st.error("Data not available.")
    else:
        df = pd.read_csv(file_path, parse_dates=["date"])
        df = df[df["date"].dt.month == int(month)]
        df["day"] = df["date"].dt.day

        st.write("Data Preview:")
        st.dataframe(df.head())
        st.write("Available days:", sorted(df['day'].unique()))

        days = sorted(df["day"].unique())
        selected_day = st.sidebar.selectbox("Select Day", days)
        day_df = df[df["day"] == selected_day]

        st.subheader(f"{model.upper()} - {region.title()} on {month_name} {selected_day}, 2024")

        if day_df.empty:
            st.warning("⚠️ No data available for this day.")
        else:
            fig = px.line(day_df, x="hour", y="predicted_temperature", title="Predicted Temperature")
            if "actual_temperature" in day_df.columns and st.checkbox("Show Actual Temperature"):
                fig.add_scatter(x=day_df["hour"], y=day_df["actual_temperature"], name="Actual", mode='lines')
            st.plotly_chart(fig, use_container_width=True)

# ========================================
# 📊 VIEW 2: Actual vs Predicted (2024)
# ========================================
elif view == "📊 Actual vs Predicted (2024)":
    selected_day = st.sidebar.number_input("Select Day of Month", min_value=1, max_value=31, step=1)
    st.subheader(f"📊 Comparison on {region.title()}, {month_name} {selected_day}, 2024")

    rmse_data = []

    for model in MODELS:
        file_path = f"{DATA_DIR}/{model}_{region}_2024.csv"
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path, parse_dates=["date"])
        df = df[df["date"].dt.month == int(month)]
        df = df[df["date"].dt.day == selected_day]

        if df.empty or "actual_temperature" not in df.columns:
            continue

        fig = px.line(df, x="hour", y="predicted_temperature", title=model.upper())
        fig.add_scatter(x=df["hour"], y=df["actual_temperature"], name="Actual", mode='lines')
        st.plotly_chart(fig, use_container_width=True)

        rmse = np.sqrt(np.mean((df["actual_temperature"] - df["predicted_temperature"]) ** 2))
        rmse_data.append((model, round(rmse, 3)))

    if rmse_data:
        st.markdown("### 📋 RMSE Table")
        rmse_df = pd.DataFrame(rmse_data, columns=["Model", "RMSE (°C)"])
        st.dataframe(rmse_df)

# ========================================
# 🔮 VIEW 3: 2025 Unseen Forecast
# ========================================
elif view == "🔮 2025 Unseen Forecast":
    model = st.sidebar.selectbox("Select Model", MODELS)
    file_path = f"{DATA_DIR}/{model}_{region}_2025.csv"

    if not os.path.exists(file_path):
        st.error("Data not available.")
    else:
        df = pd.read_csv(file_path, parse_dates=["date"])
        df = df[df["date"].dt.month == int(month)]
        df["day"] = df["date"].dt.day

        days = sorted(df["day"].unique())
        selected_day = st.sidebar.selectbox("Select Day", days)
        day_df = df[df["day"] == selected_day]

        st.subheader(f"{model.upper()} Forecast - {region.title()} on {month_name} {selected_day}, 2025")
        if day_df.empty:
            st.warning("⚠️ No data available for this day.")
        else:
            fig = px.line(day_df, x="hour", y="predicted_temperature", title="Predicted Temperature")
            st.plotly_chart(fig, use_container_width=True)

        if st.button("📊 Compare Models"):
            st.markdown("## 📊 Model Comparison")

            for m in MODELS:
                m_path = f"{DATA_DIR}/{m}_{region}_2025.csv"
                if not os.path.exists(m_path):
                    continue
                m_df = pd.read_csv(m_path, parse_dates=["date"])
                m_df = m_df[(m_df["date"].dt.month == int(month)) & (m_df["date"].dt.day == selected_day)]
                if m_df.empty:
                    continue

                fig = px.line(m_df, x="hour", y="predicted_temperature", title=f"{m.upper()} Prediction")
                st.plotly_chart(fig, use_container_width=True)

            st.info("✅ No actual data for 2025, so comparison is prediction-only.")
