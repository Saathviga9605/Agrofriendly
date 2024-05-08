import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("dataset.csv")

st.sidebar.title("Predict the price of your produce!")

crop = st.sidebar.selectbox("Select your Crop", sorted(data['Commodity'].unique()))

filtered_data = data[data['Commodity'] == crop]

st.title("Crop Price Prediction - Team AgroFriends")

if len(filtered_data) >= 2:
    X = filtered_data.drop(columns=['Modal_x0020_Price'])  
    y = filtered_data['Modal_x0020_Price']

    X_encoded = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    st.write("### Prediction")
    prediction = round(model.predict(X_test)[0], 2)
    st.write(f"<div class='card'><h2 style='text-align: center;'>Crop Price Prediction</h2><div style='text-align: center;'><h3>Predicted Price for {crop} per quintal</h3><p style='font-size: 24px;'>Rs.{prediction}</p></div></div>", unsafe_allow_html=True)

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

st.sidebar.markdown("---")
st.sidebar.write("Created with ❤️ by Team AgroFriends")
