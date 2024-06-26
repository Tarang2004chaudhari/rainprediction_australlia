from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load your pre-trained model and preprocessing tools
logicstic = joblib.load('logicstic_model.pkl')
encoder = joblib.load('encoder.pkl')
scaling_data = joblib.load('scaling_data.pkl')
data_add = joblib.load('data_add.pkl')

# Define your columns
numeric_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                   'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
                   'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 
                   'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
object_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
encoded_columns = list(encoder.get_feature_names_out(object_columns))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = {
        # 'Date': request.form['Date'],
        'Location': request.form['Location'],
        'MinTemp': float(request.form['MinTemp']),
        'MaxTemp': float(request.form['MaxTemp']),
        'Rainfall': float(request.form['Rainfall']),
        'Evaporation': float(request.form['Evaporation']),
        'Sunshine': float(request.form['Sunshine']) if request.form['Sunshine'] else np.nan,
        'WindGustDir': request.form['WindGustDir'],
        'WindGustSpeed': float(request.form['WindGustSpeed']),
        'WindDir9am': request.form['WindDir9am'],
        'WindDir3pm': request.form['WindDir3pm'],
        'WindSpeed9am': float(request.form['WindSpeed9am']),
        'WindSpeed3pm': float(request.form['WindSpeed3pm']),
        'Humidity9am': float(request.form['Humidity9am']),
        'Humidity3pm': float(request.form['Humidity3pm']),
        'Pressure9am': float(request.form['Pressure9am']),
        'Pressure3pm': float(request.form['Pressure3pm']),
        'Cloud9am': float(request.form['Cloud9am']),
        'Cloud3pm': float(request.form['Cloud3pm']),
        'Temp9am': float(request.form['Temp9am']),
        'Temp3pm': float(request.form['Temp3pm']),
        'RainToday': request.form['RainToday']
    }

    data1_input = pd.DataFrame([input_data])
    data1_input[numeric_columns] = data_add.transform(data1_input[numeric_columns])
    data1_input[numeric_columns] = scaling_data.transform(data1_input[numeric_columns])
    data1_input[encoded_columns] = encoder.transform(data1_input[object_columns])
    
    # # Debugging statements to check processed data
    # print("Processed Input Data:")
    # print(data1_input[numeric_columns + encoded_columns])

    pred = logicstic.predict(data1_input[numeric_columns + encoded_columns])[0]
    prob = logicstic.predict_proba(data1_input[numeric_columns + encoded_columns])[0][list(logicstic.classes_).index(pred)]

    return render_template('result.html', prediction=pred, probability=prob)

if __name__ == '__main__':
    app.run(debug=True)
