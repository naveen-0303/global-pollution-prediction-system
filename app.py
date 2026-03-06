import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, date, timedelta
import io
import requests
import pydeck as pdk

st.set_page_config(page_title="Global Pollution Prediction System", page_icon="🌍", layout="wide")

# ========================= LOAD DATA =========================
@st.cache_data
def load_data():
    return pd.read_excel("global_dataset_with_coordinates.xlsx")

df = load_data()
df.columns = df.columns.str.lower()   # normalize names

@st.cache_resource
def load_model():
    return joblib.load("pollution_model.pkl")

model = load_model()

# ========================= HELPERS =========================
HISTORY_FILE = "prediction_history.csv"

def save_history(record: dict):
    pd.DataFrame([record]).to_csv(HISTORY_FILE, mode="a",
                                  header=not os.path.exists(HISTORY_FILE), index=False)

def load_history():
    return pd.read_csv(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else pd.DataFrame()

def create_pdf_report(info):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    y = A4[1] - 50
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Pollution Prediction Report")
    y -= 40
    c.setFont("Helvetica", 12)
    for k, v in info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 18
        if y < 60: c.showPage(); y = A4[1] - 60
    c.save(); buffer.seek(0)
    return buffer.getvalue()

def get_live_aqi(city):
    try:
        url = f"https://api.waqi.info/feed/{city}/?token=demo"
        data = requests.get(url, timeout=8).json()
        if data.get("status") != "ok": return None, "No data found"
        aqi = data["data"].get("aqi")
        pm25 = data["data"].get("iaqi", {}).get("pm25", {}).get("v")
        return {"AQI": aqi, "PM2.5": pm25}, None
    except Exception as e:
        return None, str(e)

# ========================= SIDEBAR =========================
with st.sidebar:
    st.markdown("## 🌍 About")
    st.write("AI model trained on global air & water pollution indicators.")
    st.write("Developed by **Naveen & Ramya**")
    st.markdown("---")
    st.write("📂 Files in use")
    st.write("✔ global_dataset_with_coordinates.xlsx")
    st.write("✔ pollution_model.pkl")

# ========================= MAIN =========================
st.title("🌍 Global Pollution Prediction System")
st.caption("AI-Based Pollution Risk Analysis, Visualization & Forecasting")
st.markdown("---")

tab_predict, tab_forecast, tab_map, tab_history, tab_live, tab_city = st.tabs(
    ["🧮 Prediction", "📈 7-Day Forecast", "🗺 Map", "📚 History & Report", "🌐 Live AQI", "🏙 City Lookup"]
)



# -------------------------------------------------
#  TAB 1: PREDICTION
# -------------------------------------------------
with tab_predict:
    st.subheader("🧮 Enter Environmental Parameters")

    col1, col2 = st.columns(2)
    with col1:
        air_q = st.number_input("Air Quality Index", 0.0, 100.0, 50.0)
        who_pm25 = st.number_input("WHO PM2.5 Score", 0.0, 50.0, 10.0)
    with col2:
        water_p = st.number_input("Water Pollution Index", 0.0, 100.0, 60.0)
        wb_score = st.number_input("WB Water Pollution Score", 0.0, 50.0, 5.0)

    if st.button("🔮 Predict Pollution Risk", use_container_width=True):

        # ------- MODEL PREDICTION -------
        X = np.array([[air_q, who_pm25, water_p, wb_score]])
        combined = float(model.predict(X)[0])
        combined = round(max(0, min(1, combined)), 3)

        # ------- CATEGORY -------
        if combined >= 0.75: icon, cat = "🔴", "Extreme Risk"
        elif combined >= 0.60: icon, cat = "🟥", "High Risk"
        elif combined >= 0.40: icon, cat = "🟧", "Moderate Risk"
        else: icon, cat = "🟩", "Low Risk"

        # ------- NEAREST SIMILAR CITY -------
        df["dist"] = np.sqrt((df["airquality"] - air_q)**2 + (df["waterpollution"] - water_p)**2)
        near = df.sort_values("dist").iloc[0]
        near_city, near_country = near["city"], near["country"]

        # ------- 3 METRICS SIDE-BY-SIDE -------
        c1, c2, c3 = st.columns(3)
        c1.metric("Combined Score", f"{combined:.3f}")
        c2.metric("Risk Category", f"{icon} {cat}")
        c3.metric("Nearest Similar City", f"{near_city}, {near_country}")

        # ------- PROGRESS BAR -------
        st.progress(int(combined * 100))

        # ------- CHARTS SIDE BY SIDE -------
        b1, b2 = st.columns(2)

        air_norm = air_q / 100
        water_norm = water_p / 100
        df_chart = pd.DataFrame(
            {"Contribution": [air_norm, water_norm]},
            index=["Air Quality", "Water Pollution"]
        )

        with b1:
            st.subheader("📊 Contribution Bar Chart")
            st.bar_chart(df_chart)

        with b2:
            st.subheader("🥧 Contribution Pie Chart")
            fig, ax = plt.subplots()
            ax.pie(
                [air_norm, water_norm],
                labels=["Air Quality", "Water Pollution"],
                autopct="%1.1f%%"
            )
            st.pyplot(fig)

        # ------- INTERPRETATION -------
        interp = (
            f"The model predicts **{cat}** with a combined score of **{combined:.2f}**. "
            f"Air quality contributes **{air_norm/(air_norm+water_norm+1e-9)*100:.1f}%** and "
            f"water pollution contributes **{water_norm/(air_norm+water_norm+1e-9)*100:.1f}%** "
            f"to the risk level. The most similar pollution profile globally is "
            f"**{near_city}, {near_country}**."
        )
        st.markdown("📝 " + interp)

        # ------- SAVE TO SESSION + HISTORY -------
        rec = {
            "air_q": air_q, "who_pm25": who_pm25, "water_p": water_p, "wb_score": wb_score,
            "combined_score": combined, "risk_category": cat,
            "nearest_city": near_city, "nearest_country": near_country,
            "interpretation": interp, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state["last_prediction"] = rec
        save_history(rec)

# -------------------------------------------------
# TAB 2: FORECAST
# -------------------------------------------------
with tab_forecast:
    if "last_prediction" not in st.session_state:
        st.info("Run a prediction first.")
    else:
        base = st.session_state["last_prediction"]["combined_score"]
        days = [date.today() + timedelta(i) for i in range(7)]
        values = [max(0, min(1, base + np.random.uniform(-0.03, 0.03))) for _ in days]
        st.line_chart(pd.DataFrame({"Date": days, "Score": values}).set_index("Date"))

# -------------------------------------------------
# TAB 3: MAP
# -------------------------------------------------
with tab_map:
    if {"latitude", "longitude"}.issubset(df.columns):
        df["risk_level"] = df["combinedpollutionscore"]
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer(
                "ScatterplotLayer", data=df,
                get_position=["longitude", "latitude"],
                get_radius=19000,
                get_fill_color="[risk_level * 255, 50, (1-risk_level) * 255, 200]",
                pickable=True)],
            initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1),
            tooltip={"text": "{city}, {country}\nRisk: {risk_level}"}
        ))
    else:
        st.error("Latitude/longitude missing in dataset")

# -------------------------------------------------
# TAB 4: HISTORY + PDF
# -------------------------------------------------
with tab_history:
    hist = load_history()
    st.dataframe(hist.tail(20))

    if "last_prediction" in st.session_state:
        pdf = create_pdf_report(st.session_state["last_prediction"])
        st.download_button("⬇ Download PDF Report", pdf, file_name="report.pdf")

# -------------------------------------------------
#  TAB 5: LIVE AQI (WAQI Real-Time)
# -------------------------------------------------
with tab_live:
    st.subheader("🌐 Live Air Quality (PM2.5) — WAQI Real-Time")

    city = st.text_input("Enter city name (Example: Delhi, London, Chennai, Paris):")

    WAQI_TOKEN = "3ed72b6d6cff9e57d3992bc2d9aaa0e9469058f9"

    if st.button("🔄 Fetch Live AQI", use_container_width=True):
        if not city.strip():
            st.warning("⚠ Please enter a city name.")
        else:
            try:
                url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
                response = requests.get(url, timeout=10).json()

                if response.get("status") != "ok":
                    st.error("❌ No live monitoring station found for this city.")
                else:
                    aqi = response["data"].get("aqi", "N/A")

                    pm25 = None
                    if "iaqi" in response["data"] and "pm25" in response["data"]["iaqi"]:
                        pm25 = response["data"]["iaqi"]["pm25"]["v"]

                    st.success(f"🌍 City: **{city.title()}**")
                    st.metric("Live AQI", aqi)

                    if pm25 is not None:
                        st.metric("PM2.5 (µg/m³)", pm25)
                    else:
                        st.warning("⚠ PM2.5 value not available in this station.")

                    st.caption("📌 Source: WAQI (World Air Quality Index) — Real-Time API")
            except Exception as e:
                st.error(f"⚠ API Error: {str(e)}")

# -------------------------------------------------
#  TAB 6: CITY LOOKUP
# -------------------------------------------------
with tab_city:
    st.subheader("🏙 City Lookup – Detailed Pollution Profile")

    city_input = st.text_input("Enter a city name (example: Delhi, Chennai, London, Tokyo):")

    if st.button("🔍 Search City", use_container_width=True):
        if not city_input.strip():
            st.warning("Please enter a city name.")
        else:
            row = df[df["city"].str.lower() == city_input.lower().strip()]

            if row.empty:
                st.error("❌ City not found in dataset.")
            else:
                row = row.iloc[0]

                # Extract values
                air_q = row["airquality"]
                who_pm25 = row["who_pm25"]
                water_p = row["waterpollution"]
                wb_score = row["wb_waterpollutionscore"]
                combined = row["combinedpollutionscore"]
                category = row["pollutionriskcategory"]
                country = row["country"]
                lat = row["latitude"]
                lon = row["longitude"]

                # ---------------- Metrics (side-by-side)
                m1, m2, m3 = st.columns(3)
                m1.metric("Air Quality Index", f"{air_q}")
                m2.metric("Water Pollution Index", f"{water_p}")
                m3.metric("Combined Pollution Score", f"{combined:.2f}")

                st.metric("Risk Category", category)

                st.markdown("---")

                # ---------------- Bar Chart + Pie Chart
                st.subheader("📊 Pollution Contribution Charts")
                c1, c2 = st.columns(2)

                air_norm = air_q / 100
                water_norm = water_p / 100
                contrib_df = pd.DataFrame(
                    {"Contribution": [air_norm, water_norm]},
                    index=["Air Quality", "Water Pollution"]
                )

                with c1:
                    st.bar_chart(contrib_df)

                with c2:
                    fig, ax = plt.subplots()
                    ax.pie([air_norm, water_norm],
                           labels=["Air Quality", "Water Pollution"],
                           autopct="%1.1f%%")
                    ax.set_title("Relative Contribution")
                    st.pyplot(fig)

                # ---------------- Interpretation
                st.markdown("### 📝 Interpretation")
                interpretation = (
                    f"City **{city_input.title()}**, {country.title()} currently falls under **{category}** risk "
                    f"with a pollution score of **{combined:.2f}**. "
                    f"Air quality contributes **{air_norm/(air_norm+water_norm)*100:.1f}%**, while "
                    f"water pollution contributes **{water_norm/(air_norm+water_norm)*100:.1f}%** to overall risk."
                )
                st.write(interpretation)

                st.markdown("---")

                # ---------------- City Map
                st.subheader("📍 Exact City Location on Map")

                try:
                    lat = float(lat)
                    lon = float(lon)

                    st.pydeck_chart(pdk.Deck(
                        layers=[
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=[{"lat": lat, "lon": lon}],
                                get_position=["lon", "lat"],
                                get_radius=25000,
                                get_fill_color=[255, 0, 0, 200],
                                pickable=True
                            )
                        ],
                        initial_view_state=pdk.ViewState(
                            latitude=lat,
                            longitude=lon,
                            zoom=8,
                            pitch=0
                        ),
                        tooltip={"text": f"{city_input.title()}, {country.title()}"}
                    ))
                except:
                    st.error("⚠ Could not display map for this city.")

                st.markdown("---")

                # ---------------- Export Options
                st.subheader("⬇ Export City Report")

                pdf_bytes = create_pdf_report({
                    "City": city_input.title(),
                    "Country": country.title(),
                    "Air Quality Index": air_q,
                    "WHO PM2.5 Score": who_pm25,
                    "Water Pollution Index": water_p,
                    "WB Water Score": wb_score,
                    "Combined Score": combined,
                    "Risk Category": category,
                    "Latitude": lat,
                    "Longitude": lon,
                    "Interpretation": interpretation
                })

                st.download_button(
                    label="📄 Download as PDF",
                    data=pdf_bytes,
                    file_name=f"{city_input}_report.pdf",
                    mime="application/pdf"
                )

                st.download_button(
                    label="📑 Download as CSV",
                    data=row.to_csv().encode("utf-8"),
                    file_name=f"{city_input}_details.csv",
                    mime="text/csv"
                )



st.markdown("---")
st.caption("Developed by Ramya & Naveen • Global AI Pollution System")
