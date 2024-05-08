import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st


def trainModel():
    crop_data = pd.read_csv("Crop_recommendation.csv")
    X = crop_data.drop('label', axis=1)
    y = crop_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return model


def get_recommendation(model):
    st.title("Crop Recommendation System")
    n = st.number_input("Nitrogen content (N):")
    p = st.number_input("Phosphorus content (P):")
    k = st.number_input("Potassium content (K):")
    temperature = st.number_input("Temperature (°C):")
    humidity = st.number_input("Humidity (%):")
    ph = st.number_input("pH value:")
    rainfall = st.number_input("Rainfall (mm):")
    if st.button("Predict Crop", key="predict_button"):
        user_input = pd.DataFrame({
            'N': [n],
            'P': [p],
            'K': [k],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })

        recommended_crop = model.predict(user_input)[0]
        st.write(f"<div class='card'><div style='text-align: center;'><h3>Recommended crop</h3><p style='font-size: 24px;'>{recommended_crop}</p></div></div>", unsafe_allow_html=True)

model = trainModel() 
get_recommendation(model) 
st.markdown("""<style>.st-button {text-align: center;}</style>""", unsafe_allow_html=True)
