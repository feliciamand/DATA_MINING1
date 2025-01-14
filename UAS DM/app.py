import streamlit as st
import pickle
import numpy as np

# Memuat model dan scaler
with open('kmeans_model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Judul aplikasi
st.title("Prediksi Segmentasi Pelanggan")

# Input pengguna
gender = st.selectbox("Jenis Kelamin:", ["Pria", "Wanita"])
age = st.slider("Usia:", 18, 70, 30)
annual_income = st.number_input("Pendapatan Tahunan (k$):", min_value=10, max_value=150, value=50)
spending_score = st.slider("Skor Belanja (1-100):", 1, 100, 50)

# Konversi input menjadi array
gender_encoded = 1 if gender == "Pria" else 0
user_input = np.array([[gender_encoded, age, annual_income, spending_score]])

# Normalisasi input
scaled_input = scaler.transform(user_input)

# Prediksi cluster
cluster = kmeans.predict(scaled_input)[0]

# Tampilkan hasil
st.write(f"Pelanggan termasuk ke dalam cluster: {cluster}")

