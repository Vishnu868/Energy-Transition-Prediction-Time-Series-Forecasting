"""EnergyTripAI - Streamlit dashboard for load forecasting and breaker-trip prediction.

Run:  streamlit run app/energytrip_app.py
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models import arima_forecast, sarima_forecast  # noqa: E402

st.set_page_config(page_title="EnergyTripAI", layout="wide")
st.title("⚡ EnergyTripAI — Electricity Load Forecasting & Trip Prediction")

with st.sidebar:
    st.header("System configuration")
    breaker = st.number_input("Breaker rating (A)", 1.0, 200.0, 15.0)
    voltage = st.number_input("System voltage (V)", 100.0, 440.0, 230.0)
    location = st.selectbox("Location type", ["Residential", "Commercial", "Industrial"])
    freq = st.selectbox("Resampling frequency", ["h", "15min", "D"], index=0)
    horizon = st.slider("Forecast horizon (steps)", 6, 168, 24)
    model_name = st.selectbox("Model", ["SARIMA", "ARIMA"])

p_max = voltage * breaker
p_warn = 0.8 * p_max

file = st.file_uploader("Upload electricity load CSV (timestamp + power in W)", type="csv")
if not file:
    st.info(f"Max allowable power: **{p_max:.0f} W** · Warning threshold: **{p_warn:.0f} W**")
    st.stop()

df = pd.read_csv(file)
time_col = st.selectbox("Timestamp column", df.columns)
load_col = st.selectbox("Power column (W)", [c for c in df.columns if c != time_col])
s = (df.assign(**{time_col: pd.to_datetime(df[time_col])})
       .set_index(time_col)[load_col].astype(float).resample(freq).mean().interpolate())
st.caption(f"{len(df):,} records loaded · {len(s):,} points after resampling")

with st.spinner(f"Running {model_name}..."):
    fc = (sarima_forecast if model_name == "SARIMA" else arima_forecast)(s, horizon)
idx = pd.date_range(s.index[-1], periods=horizon + 1, freq=freq)[1:]

current = s.iloc[-1]
states = np.where(fc >= p_max, "Trip", np.where(fc >= p_warn, "Warning", "Normal"))
trip_prob = float((states == "Trip").mean())
stability = max(0.0, 100 * (1 - fc.max() / p_max)) if fc.max() < p_max else 0.0
status = "TRIP RISK" if trip_prob > 0 else ("WARNING" if (states == "Warning").any() else "NORMAL")

c = st.columns(5)
c[0].metric("System status", status)
c[1].metric("Current power", f"{current:.0f} W")
c[2].metric("Current", f"{current / voltage:.2f} A")
c[3].metric("Load %", f"{100 * current / p_max:.1f}%")
c[4].metric("Trip probability", f"{trip_prob:.0%}", help=f"Stability score: {stability:.0f}/100")

fig = go.Figure()
fig.add_scatter(x=s.index[-500:], y=s.values[-500:], name="Historical", line=dict(color="royalblue"))
fig.add_scatter(x=idx, y=fc, name=f"{model_name} forecast", line=dict(color="crimson"))
fig.add_hline(y=p_warn, line_dash="dash", line_color="orange", annotation_text="Warning (80%)")
fig.add_hline(y=p_max, line_dash="dash", line_color="red", annotation_text="Trip threshold")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Predicted system state probability")
st.bar_chart(pd.Series(states).value_counts(normalize=True))
