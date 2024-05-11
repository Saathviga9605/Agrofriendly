import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Function to train the crop recommendation model
def train_recommendation_model():
    crop_data = pd.read_csv("Crop_recommendation.csv")
    X = crop_data.drop('label', axis=1)
    y = crop_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to get crop recommendation
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
            'N': [n], 'P': [p], 'K': [k], 'temperature': [temperature],
            'humidity': [humidity], 'ph': [ph], 'rainfall': [rainfall]
        })
        recommended_crop = model.predict(user_input)[0]
        st.write(f"<div class='card'><div style='text-align: center;'>"
                 f"<h3>Recommended crop</h3><p style='font-size: 24px;'>{recommended_crop}</p>"
                 f"</div></div>", unsafe_allow_html=True)

# Function to train the crop yield prediction model
def train_yield_model():
    data = pd.read_csv("crop_yield.csv")
    data.dropna(inplace=True)
    X = data[['Crop', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
    y = data['Yield']
    X = pd.get_dummies(X, columns=['Crop'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model, X

# Function to predict crop yield
def predict_yield(model, X):
    st.title("Crop Yield Prediction")
    data = pd.read_csv("crop_yield.csv")
    crop = st.selectbox("Crop", data['Crop'].unique())
    area = st.number_input("Area (in hectares)", min_value=0.1, value=1.0)
    annual_rainfall = st.number_input("Annual Rainfall (in mm)", min_value=0, value=500)
    fertilizer = st.number_input("Fertilizer Used (in kilograms)", min_value=0, value=100)
    pesticide = st.number_input("Pesticide Used (in kilograms)", min_value=0, value=10)
    # Prediction
    input_data = {
        'Crop': crop, 'Area': area, 'Annual_Rainfall': annual_rainfall,
        'Fertilizer': fertilizer, 'Pesticide': pesticide
    }
    input_df = pd.DataFrame([input_data])
    if st.button("Predict"):
        input_df = pd.get_dummies(input_df, columns=['Crop'])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)
        prediction = model.predict(input_df)
        st.subheader("Prediction")
        st.write("Predicted Yield:", prediction[0], " kilograms/hectare")

# Function to train the crop price prediction model
def train_price_model():
    data = pd.read_csv("dataset.csv")
    return data

# Function to predict crop price
def predict_price(data):
    st.title("Crop Price Prediction")
    crop = st.selectbox("Select your Crop", sorted(data['Commodity'].unique()))
    filtered_data = data[data['Commodity'] == crop]
    if len(filtered_data) >= 2:
        X = filtered_data.drop(columns=['Modal_x0020_Price'])
        y = filtered_data['Modal_x0020_Price']
        X_encoded = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        prediction = round(model.predict(X_test)[0], 2)
        st.write(f"<div class='card'><h2 style='text-align: center;'>Crop Price Prediction</h2><div style='text-align: center;'>"
                 f"<h3>Predicted Price for {crop} per quintal</h3><p style='font-size: 24px;'>Rs.{prediction}</p></div></div>", unsafe_allow_html=True)
        st.write("#### Dataset")
        st.write(filtered_data.head())
        st.write("### Data Visualization")
        st.write("#### Price Distribution")
        fig, ax = plt.subplots()
        sns.barplot(data=filtered_data.head(), x='State', y='Modal_x0020_Price', hue='Market', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("Insufficient data for model training and evaluation.")

def train_weather_model():
    data = pd.read_csv("weather.csv")
    X = data.drop('weather', axis=1)
    y = data['weather']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to predict weather
def predict_weather(model):
    st.title("Weather Prediction")
    st.write("Enter the following details to predict the weather:")
    n = st.number_input("Precipitation:")
    p = st.number_input("Maximum Temp:")
    k = st.number_input("Minimum Temp:")
    temperature = st.number_input("Wind:")
    if st.button("Predict"):
        user_input = pd.DataFrame({
            'precipitation': [n], 'temp_max': [p], 'temp_min': [k], 'wind': [temperature]
        })
        weather = model.predict(user_input)
        st.write("Predicted Weather:", weather[0])

print("Crop Recommendation Model Training...")
model_recommendation = train_recommendation_model()
print("Crop Yield Prediction Model Training...")
model_yield, X_yield = train_yield_model()
print("Crop Price Prediction Model Training...")
data_price = train_price_model()
print("Weather Prediction Model Training...")
model_weather = train_weather_model()
print("All Model Training Over... Server is ready to Serve")


