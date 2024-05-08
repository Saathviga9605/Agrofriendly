import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv("crop_yield.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Split the dataset into input features (X) and target variable (y)
X = data[['Crop', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = data['Yield']

# One-hot encode the 'Crop' column
X = pd.get_dummies(X, columns=['Crop'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Streamlit app
st.title("Crop Yield Prediction")

# Input fields
crop = st.selectbox("Crop", data['Crop'].unique())
area = st.number_input("Area (in hectares)", min_value=0.1, value=1.0)
annual_rainfall = st.number_input("Annual Rainfall (in mm)", min_value=0, value=500)
fertilizer = st.number_input("Fertilizer Used (in kilograms)", min_value=0, value=100)
pesticide = st.number_input("Pesticide Used (in kilograms)", min_value=0, value=10)

# Prediction
input_data = {
    'Crop': crop,
    'Area': area,
    'Annual_Rainfall': annual_rainfall,
    'Fertilizer': fertilizer,
    'Pesticide': pesticide
}

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    # One-hot encode the input crop
    input_df = pd.get_dummies(input_df, columns=['Crop'])
    # Ensure input DataFrame has same columns as training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(input_df)
    st.subheader("Prediction")
    st.write("Predicted Yield:", prediction[0], " kilograms/hectare")
