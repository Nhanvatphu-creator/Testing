import streamlit as st
import joblib

# Load mô hình
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Shopee Sentiment Analysis")

user_input = st.text_area("Nhập review:")
if st.button("Phân tích"):
    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]
    st.write("Kết quả:", "Tích cực" if prediction == 1 else "Tiêu cực")
