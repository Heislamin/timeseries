import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import glob

# --- CONFIGURATION ---
DATA_DIR = "csvs_extracted/data"

# Auto-detect all model names from any CSV in the folder
def detect_models(data_dir):
    files = glob.glob(f"{data_dir}/*_*.csv")
    models = [os.path.basename(f).split("_")[0] for f in files]
    return sorted(list(set(models)))

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
view = st.sidebar.radio("Select View", ["📅 2024 Forecast"])

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

    st.write(f"📂 File path used: `{file_path}`")

    if not os.path.exists(file_path):
        st.error("❌ File does not exist.")
    else:
        df = pd.read_csv(file_path)
        st.write("✅ File loaded. First 5 rows:")
        st.dataframe(df.head())

        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            st.error(f"❌ Date parsing failed: {e}")
        else:
            st.success("✅ Date column parsed successfully.")

        df = df[df["date"].dt.month == int(month)]
        df["day"] = df["date"].dt.day
        st.write("📅 Unique days in selected month:", sorted(df['day'].unique()))

        days = sorted(df["day"].unique())
        if len(days) == 0:
            st.warning("⚠️ No data found for selected month.")
        else:
            selected_day = st.sidebar.selectbox("Select Day", days)
            day_df = df[df["day"] == selected_day]

            st.subheader(f"{model.upper()} - {region.title()} on {month_name} {selected_day}, 2024")

            if day_df.empty:
                st.warning("⚠️ No data available for this day.")
            else:
                st.write("📈 Daily Data Preview:")
                st.dataframe(day_df.head())
                fig = px.line(day_df, x="hour", y="predicted_temperature", title="Predicted Temperature")
                if "actual_temperature" in day_df.columns and st.checkbox("Show Actual Temperature"):
                    fig.add_scatter(x=day_df["hour"], y=day_df["actual_temperature"], name="Actual", mode='lines')
                st.plotly_chart(fig, use_container_width=True)
