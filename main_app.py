import streamlit as st
from streamlit_option_menu import option_menu
import model_training as model

# Run the main Streamlit app
if __name__ == "__main__":
   
    st.sidebar.image("logo.png")
    with st.sidebar:
        nav_option = option_menu(
            menu_title="Agrofriendly",
            options=['Home', 'Get Crop Recommendations', 'Predict Crop Yield', 'Price Prediction', 'Weather Prediction'],
            icons= ['house', 'hand-thumbs-up', 'sign-yield', 'cash-coin', 'cloud-sun'], 
            menu_icon= 'cast', 
            default_index= 0,
        )
    if nav_option == 'Home':
        st.title("Welcome to AgroFriendly!!!")
        st.write("AgroFriendly is your agricultural companion, providing recommendations for crop, predicting the yield of your produce, determine the market price of your harvest and predict the weather conditions too!!!")
        st.write("Select an option from the navigation bar to get started.")
 
    elif nav_option == 'Get Crop Recommendations':
        model.get_recommendation(model.model_recommendation)

    elif nav_option == 'Predict Crop Yield':
        model.predict_yield(model.model_yield, model.X_yield)

    elif nav_option == 'Price Prediction':
        model.predict_price(model.data_price)

    elif nav_option == 'Weather Prediction':
        model.predict_weather(model.model_weather)
