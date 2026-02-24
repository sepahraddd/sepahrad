from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import requests
import numpy as np

app = Flask(__name__)
CORS(app)

# لود کردن مدل - فایل باید کنار همین کد باشه
model = keras.models.load_model("dnn_model.keras")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        time_input = data.get('time', '23:30')
        hour = int(time_input.split(':')[0])

        # مختصات کرج طبق فایل خودت
        latitude, longitude = 35.8400, 50.9391
        date = "2026-02-13" 

        # فراخوانی API هواشناسی مشابه کد خودت
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
               f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m"
               f"&start={date}T00:00:00Z&end={date}T23:59:00Z&timezone=Asia/Tehran")
        
        response = requests.get(url).json()
        
        temp = response["hourly"]["temperature_2m"][hour]
        wind_speed = response["hourly"]["windspeed_10m"][hour]
        humidity = response["hourly"]["relative_humidity_2m"][hour]

        # مقادیر ثابت طبق فایل اصلی تو
        inputs = np.array([[temp, wind_speed, humidity, 20.5, 25.12]], dtype=np.float32)
        
        # پیش‌بینی
        prediction = model.predict(inputs)
        seeing_value = float(prediction[0][0])

        return jsonify({
            "status": "success",
            "seeing": seeing_value,
            "temp": temp,
            "humidity": humidity
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print("Server is running on port 5000...")
    import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)