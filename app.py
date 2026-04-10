from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model dan file pendukung
model = load_model("model_diabetes.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset asli untuk grafik tren
df = pd.read_csv("diabetes_jabar.csv")

# Daftar kabupaten/kota untuk dropdown
kabupaten_list = sorted(label_encoder.classes_.tolist())


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    selected_kabupaten = None
    selected_tahun = None
    chart_labels = []
    chart_values = []

    if request.method == "POST":
        try:
            selected_kabupaten = request.form["kabupaten"]
            selected_tahun = int(request.form["tahun"])

            # Encode kabupaten
            kabupaten_encoded = label_encoder.transform([selected_kabupaten])[0]

            # Prediksi dengan model ANN
            input_data = np.array([[kabupaten_encoded, selected_tahun]])
            input_scaled = scaler_X.transform(input_data)

            pred_scaled = model.predict(input_scaled, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]

            # Supaya tidak negatif
            prediction = max(0, round(pred))

            # Data tren kabupaten terpilih
            df_kab = df[df["nama_kabupaten_kota"] == selected_kabupaten].copy()
            df_kab = df_kab.sort_values("tahun")

            chart_labels = df_kab["tahun"].tolist()
            chart_values = df_kab["jumlah_penderita_dm"].tolist()

        except Exception as e:
            error = f"Terjadi kesalahan: {str(e)}"

    return render_template(
        "index.html",
        kabupaten_list=kabupaten_list,
        prediction=prediction,
        error=error,
        selected_kabupaten=selected_kabupaten,
        selected_tahun=selected_tahun,
        chart_labels=json.dumps(chart_labels),
        chart_values=json.dumps(chart_values)
    )


if __name__ == "__main__":
    app.run(debug=True)