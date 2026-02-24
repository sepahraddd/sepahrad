import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# مسیر مدل در گیت‌هاب تو
model_path = os.path.join(os.getcwd(), 'dnn_model.keras')
model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        time_input = data.get('time', '20:00')
        hour = int(time_input.split(':')[0])

        # مختصات کرج (طبق کدهای قبلی‌ات)
        lat, lon = 35.8400, 50.9391
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,windspeed_10m&timezone=Asia/Tehran"
        
        response = requests.get(url).json()
        temp = response["hourly"]["temperature_2m"][hour]
        wind = response["hourly"]["windspeed_10m"][hour]
        hum = response["hourly"]["relative_humidity_2m"][hour]

        # ورودی برای مدل هوش مصنوعی
        inputs = np.array([[temp, wind, hum, 20.0, 25.0]], dtype=np.float32)
        prediction = model.predict(inputs)
        seeing_value = float(prediction[0][0])

        return jsonify({
            "status": "success",
            "seeing": seeing_value,
            "temp": temp,
            "humidity": hum
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
