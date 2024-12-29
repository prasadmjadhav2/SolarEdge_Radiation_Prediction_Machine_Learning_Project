from flask import Flask, request, render_template
import numpy as np
import xgboost as xgb
import pickle

# Load the XGBoost model
model = pickle.load(open('x_solar_predictor.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def solar_prediction():
  return render_template('solar_prediction.html')

@app.route('/predict', methods=['POST'])
def predict_radiation():
  if request.method == 'POST':
    unixtime = int(request.form['unixtime'])
    temperature = int(request.form['temperature'])
    pressure = float(request.form['pressure'])
    humidity = int(request.form['humidity'])
    winddirection_dgr = float(request.form['winddirection_dgr'])
    speed = float(request.form['speed'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    hour = int(request.form['hour'])
    minute = int(request.form['minute'])
    second = int(request.form['second'])
    sunrise_minute = int(request.form['sunrise_minute'])
    sunset_hour = int(request.form['sunset_hour'])
    sunset_minute = int(request.form['sunset_minute'])

    input_point = np.array([[unixtime, temperature, pressure, humidity, winddirection_dgr, speed, month, day, hour, minute, second, sunrise_minute, sunset_hour, sunset_minute]])
    dinput_point = xgb.DMatrix(input_point)

    if model is not None:
      prediction = model.predict(dinput_point)
      return render_template('result.html', prediction=prediction[0])
    else:
      return "Error: Please load a trained XGBoost model for prediction."

  return "Invalid request method."

if __name__ == '__main__':
  app.run(debug=True)