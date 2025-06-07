from flask import Flask, render_template, request, jsonify
from joblib import load  # sửa ở đây
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load mô hình đã huấn luyện với joblib
model = load('model/car_price_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Lấy dữ liệu
        manufacturer = data['manufacturer']
        year = int(data['year'])
        transmission = data['transmission']
        mileage = int(data['mileage'])
        fuelType = data['fuelType']
        tax = int(data['tax']) if data['tax'] else 0
        mpg = float(data['mpg']) if data['mpg'] else 0
        engineSize = float(data['engineSize']) if data['engineSize'] else 0

        # Tạo DataFrame với tên cột CHÍNH XÁC như lúc train
        input_df = pd.DataFrame([{
            'Manufacturer': manufacturer,
            'year': year,
            'transmission': transmission,
            'mileage': mileage,
            'fuelType': fuelType,
            'tax': tax,
            'mpg': mpg,
            'engineSize': engineSize
        }])

        prediction = model.predict(input_df)[0]

        return jsonify({'predicted_price': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
