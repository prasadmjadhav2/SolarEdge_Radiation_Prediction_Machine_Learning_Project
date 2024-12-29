import streamlit as st
import numpy as np
import xgboost as xgb
import pickle

# DMatrix Xmodel
model = pickle.load(open('x_solar_predictor.pkl','rb'))

def predict_radiation(unixtime, temperature, pressure, humidity, winddirection_dgr, speed, month, day, hour, minute, second, sunrise_minute, sunset_hour, sunset_minute):
    """
    Predicts solar radiation using the trained XGBoost model.

    Args:
        unixtime (int): UNIX timestamp.
        temperature (int): Temperature in degrees Celsius.
        pressure (float): Pressure in Pascals.
        humidity (int): Humidity percentage.
        winddirection_dgr (float): Wind direction in degrees.
        speed (float): Wind speed.
        month (int): Month (1-12).
        day (int): Day of the month.
        hour (int): Hour (0-23).
        minute (int): Minute (0-59).
        second (int): Second (0-59).
        sunrise_minute (int): Sunrise minute.
        sunset_hour (int): Sunset hour.
        sunset_minute (int): Sunset minute.

    Returns:
        float: Predicted solar radiation.
    """

    input_point = np.array([[unixtime, temperature, pressure, humidity, winddirection_dgr, speed, month, day, hour, minute, second, sunrise_minute, sunset_hour, sunset_minute]])
    dinput_point = xgb.DMatrix(input_point)

    if model is not None:
        prediction = model.predict(dinput_point)
        return prediction[0]
    else:
        st.error("Please load a trained XGBoost model for prediction.")
        return None

st.title("Solar Radiation Prediction")

# Input fields for solar data
unixtime = st.number_input("UNIX Time:", min_value=0)
temperature = st.number_input("Temperature (°C):")
pressure = st.number_input("Pressure (Pa):")
humidity = st.number_input("Humidity (%):")
winddirection_dgr = st.number_input("Wind Direction (°):")
speed = st.number_input("Wind Speed:")
month = st.selectbox("Month:", range(1, 13))
day = st.number_input("Day:", min_value=1, max_value=31)
hour = st.number_input("Hour:", min_value=0, max_value=23)
minute = st.number_input("Minute:", min_value=0, max_value=59)
second = st.number_input("Second:", min_value=0, max_value=59)
sunrise_minute = st.number_input("Sunrise Minute:")
sunset_hour = st.number_input("Sunset Hour:")
sunset_minute = st.number_input("Sunset Minute:")

# Predict button
if st.button("Predict Radiation"):
    prediction = predict_radiation(unixtime, temperature, pressure, humidity, winddirection_dgr, speed, month, day, hour, minute, second, sunrise_minute, sunset_hour, sunset_minute)

    if prediction is not None:
        st.success(f"Predicted Solar Radiation: {prediction:.2f}")