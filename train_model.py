import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ======================
# 1. Load dataset
# ======================
df = pd.read_csv("diabetes_jabar.csv")

# Ambil kolom yang penting saja
df = df[["nama_kabupaten_kota", "tahun", "jumlah_penderita_dm"]]

# ======================
# 2. Encoding kabupaten
# ======================
le = LabelEncoder()
df["kabupaten_encoded"] = le.fit_transform(df["nama_kabupaten_kota"])

# ======================
# 3. Fitur dan target
# ======================
X = df[["kabupaten_encoded", "tahun"]].values
y = df["jumlah_penderita_dm"].values

# ======================
# 4. Normalisasi
# ======================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# ======================
# 5. Split data
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ======================
# 6. Model ANN
# ======================
model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ======================
# 7. Training
# ======================
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# ======================
# 8. Simpan model
# ======================
model.save("model_diabetes.h5")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
joblib.dump(le, "label_encoder.pkl")
df = df[df["jumlah_penderita_dm"] > 0]

print("Model berhasil disimpan!")