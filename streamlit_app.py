import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import glob

# --- CONFIGURATION ---
DATA_DIR = "csvs_extracted/data"

def detect_models(data_dir):
    files = glob.glob(f"{data_dir}/*_*.csv")
    models = [os.path.basename(f).split("_")[0] for f in files if "_model_metrics" not in f and "_param_" not in f]
    return sorted(list(set(models)))

MODELS = detect_models(DATA_DIR)
REGIONS = ["rakhiyal", "bopal", "ambawadi", "chandkheda", "vastral"]
MONTHS = {
    "January": "01", "February": "02", "March": "03", "April": "04",
    "May": "05", "June": "06", "July": "07", "August": "08",
    "September": "09", "October": "10", "November": "11", "December": "12"
}

st.set_page_config(page_title="Temperature Analyzer", layout="wide")
st.title("🌡️ Time Series Temperature Analyzer")

# --- SIDEBAR ---
st.sidebar.header("🔧 Filters")
view = st.sidebar.radio("Select View", ["📅 2024 Forecast", "📊 Actual vs Predicted (2024)", "🔮 2025 Unseen Forecast"])
region = st.sidebar.selectbox("Select Region", REGIONS)
month_name = st.sidebar.selectbox("Select Month", list(MONTHS.keys()))
month = MONTHS[month_name]

# ========================================
# 📅 2024 FORECAST VIEW
# ========================================
if view == "📅 2024 Forecast":
    if not MODELS:
        st.error("❌ No models detected.")
        st.stop()

    model = st.sidebar.selectbox("Select Model", MODELS, index=0)
    file_path = f"{DATA_DIR}/{model}_{region}_2024.csv"

    if not os.path.exists(file_path):
        st.error("❌ File does not exist.")
        st.stop()

    df = pd.read_csv(file_path)
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        st.error(f"❌ Date parsing failed: {e}")
        st.stop()

    df = df[df["date"].dt.month == int(month)]
    df["day"] = df["date"].dt.day
    days = sorted(df["day"].unique())
    if not days:
        st.warning("⚠️ No data for this month.")
        st.stop()

    selected_day = st.sidebar.selectbox("Select Day", days)
    day_df = df[df["day"] == selected_day]

    st.subheader(f"{model.upper()} - {region.title()} on {month_name} {selected_day}, 2024")

    if day_df.empty:
        st.warning("⚠️ No data for this day.")
    else:
        fig = px.line(day_df, x="hour", y="predicted_temperature", title="Predicted Temperature")
        if "actual_temperature" in day_df.columns and st.checkbox("Show Actual Temperature"):
            fig.add_scatter(x=day_df["hour"], y=day_df["actual_temperature"], name="Actual", mode="lines")
        st.plotly_chart(fig, use_container_width=True)

# ========================================
# 📊 ACTUAL VS PREDICTED VIEW
# ========================================
elif view == "📊 Actual vs Predicted (2024)":
    selected_day = st.sidebar.number_input("Select Day", min_value=1, max_value=31, step=1, value=10)
    st.subheader(f"📊 Actual vs Predicted - {region.title()} on {month_name} {selected_day}, 2024")

    rmse_table = []

    for model in MODELS:
        path = f"{DATA_DIR}/{model}_{region}_2024.csv"
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        try:
            df["date"] = pd.to_datetime(df["date"])
        except:
            continue

        df = df[(df["date"].dt.month == int(month)) & (df["date"].dt.day == selected_day)]

        if df.empty or "actual_temperature" not in df.columns:
            continue

        fig = px.line(df, x="hour", y="predicted_temperature", title=f"{model.upper()}")
        fig.add_scatter(x=df["hour"], y=df["actual_temperature"], name="Actual", mode="lines")
        st.plotly_chart(fig, use_container_width=True)

        rmse = np.sqrt(np.mean((df["actual_temperature"] - df["predicted_temperature"]) ** 2))
        rmse_table.append((model.upper(), round(rmse, 3)))

    if rmse_table:
        st.markdown("### 📋 RMSE Table")
        st.dataframe(pd.DataFrame(rmse_table, columns=["Model", "RMSE (°C)"]))

# ========================================
# 🔮 2025 UNSEEN FORECAST VIEW
# ========================================
elif view == "🔮 2025 Unseen Forecast":
    model = st.sidebar.selectbox("Select Model", MODELS)
    file_path = f"{DATA_DIR}/{model}_{region}_2025.csv"

    if not os.path.exists(file_path):
        st.error("❌ File does not exist.")
        st.stop()

    df = pd.read_csv(file_path)
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        st.error(f"❌ Date parsing failed: {e}")
        st.stop()

    df["day"] = df["date"].dt.day
    df = df[df["date"].dt.month == int(month)]
    days = sorted(df["day"].unique())
    if not days:
        st.warning("⚠️ No data found for this month.")
        st.stop()

    selected_day = st.sidebar.selectbox("Select Day", days)
    day_df = df[df["day"] == selected_day]

    st.subheader(f"{model.upper()} Forecast - {region.title()} on {month_name} {selected_day}, 2025")

    if day_df.empty:
        st.warning("⚠️ No data for this day.")
    else:
        fig = px.line(day_df, x="hour", y="predicted_temperature", title="Predicted Temperature (2025)")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("📊 Compare Models"):
        st.markdown("### 🔍 Model Comparison for Same Day")
        for m in MODELS:
            path = f"{DATA_DIR}/{m}_{region}_2025.csv"
            if not os.path.exists(path):
                continue

            m_df = pd.read_csv(path)
            try:
                m_df["date"] = pd.to_datetime(m_df["date"])
            except:
                continue

            m_df = m_df[(m_df["date"].dt.month == int(month)) & (m_df["date"].dt.day == selected_day)]
            if m_df.empty:
                continue

            fig = px.line(m_df, x="hour", y="predicted_temperature", title=f"{m.upper()} Prediction")
            st.plotly_chart(fig, use_container_width=True)

        st.info("✅ No actuals in 2025, so comparison is between models only.")
