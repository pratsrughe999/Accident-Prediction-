import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup

# Load the trained model and encoders
model = joblib.load('accident_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Streamlit app layout
st.title("Personal Accident Prediction in Germany")

# Dropdown inputs for user conditions
light_condition = st.selectbox("Light Conditions", ["Daylight", "Twilight", "Darkness"])
road_condition = st.selectbox("Road Surface Conditions", ["Dry", "Wet/Damp", "Slippery"])
vehicle_type = st.selectbox("Vehicle Type", ["Passenger Car", "Bicycle", "Goods Vehicle", "Other"])

# Handling unseen label for road conditions
if road_condition == "Wet/Damp":
    road_condition = "Wet"  # Map 'Wet/Damp' to 'Wet', or you could use 'Damp'

# Encode inputs using the label encoders
try:
    # Check if the input is in the encoder's known classes
    if light_condition not in label_encoders['LIGHT CONDITION'].classes_:
        raise ValueError(f"Unseen label for Light Condition: {light_condition}")

    light_condition_encoded = label_encoders['LIGHT CONDITION'].transform([light_condition])[0]

    if road_condition not in label_encoders['ROAD CONDITIONS'].classes_:
        raise ValueError(f"Unseen label for Road Condition: {road_condition}")

    road_condition_encoded = label_encoders['ROAD CONDITIONS'].transform([road_condition])[0]

except ValueError as e:
    st.error(f"Error in encoding the inputs: {e}")
    st.stop()

# Map vehicle type to features (directly encoding the vehicle type)
vehicle_features = {
    "Passenger Car": [1, 0, 0],
    "Bicycle": [0, 1, 0],
    "Goods Vehicle": [0, 0, 1],
    "Other": [0, 0, 0]
}

vehicle_encoded = vehicle_features.get(vehicle_type, [0, 0, 0])

# Prepare the input data for prediction
input_data = np.array([[
    light_condition_encoded,
    road_condition_encoded,
    *vehicle_encoded
]])

# Predict accident likelihood
if st.button("Predict Accident Likelihood"):
    prediction_proba = model.predict_proba(input_data)[0]
    likelihood = max(prediction_proba) * 100  # Get the highest probability as the likelihood

    # Decode the predicted accident severity
    predicted_severity = target_encoder.inverse_transform([np.argmax(prediction_proba)])[0]

    # Display prediction
    st.write(f"Accident Likelihood: {likelihood:.2f}%")
    st.write(f"Predicted Accident Severity: {predicted_severity}")

    # Recommendations
    if likelihood > 70:
        st.warning("High Risk! Avoid driving under these conditions if possible.")
    else:
        st.success("Conditions appear relatively safe. Stay cautious!")


# Step 4: Web Scraping Latest News Related to Road Accidents in Germany

# Function to fetch news articles related to accidents
def fetch_latest_news():
    # We will scrape multiple news sources for articles related to accidents in Germany.
    news_sources = [
        "https://www.bbc.com/news/world-europe-56585510",  # Example: BBC News
        "https://www.reuters.com/article/us-germany-accidents-idUSKBN2A10Y7",  # Example: Reuters
        "https://www.dw.com/en/germany-fatal-accident-rates-increase/a-52254838"  # Example: DW News
    ]

    articles = []

    # Loop through the URLs and scrape articles
    for url in news_sources:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "lxml")

            # Extract article title and content
            title = soup.find('h1').get_text() if soup.find('h1') else "No title found"
            description = soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {
                'name': 'description'}) else "No description found"

            # Save article if it contains relevant keywords
            if 'accident' in title.lower() or 'traffic' in title.lower():
                articles.append({
                    'title': title,
                    'description': description,
                    'url': url
                })
        except Exception as e:
            print(f"Error fetching article from {url}: {e}")

    return articles


# Fetch the latest news articles about accidents
news_articles = fetch_latest_news()

# Display the news articles
if news_articles:
    st.subheader("Latest News on Accidents in Germany")
    for article in news_articles:
        st.write(f"**Title**: {article['title']}")
        st.write(f"**Description**: {article['description']}")
        st.write(f"[Read more]({article['url']})")  # Link to the full article
else:
    st.write("No recent news articles found related to road accidents in Germany.")

# Step 5: Fetch today's weather data from WeatherAPI (Free)

# Fetch WeatherAPI key from secrets.toml
WEATHER_API_KEY = st.secrets["weather_api"]["key"]  # Fetch from Streamlit secrets
CITY_NAME = "Berlin"  # You can change this to any city in Germany

weather_url = f'http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={CITY_NAME}&aqi=no'

# Fetch the weather data
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

# Check if the API request was successful
if 'current' in weather_data:
    temperature = weather_data['current']['temp_c']
    humidity = weather_data['current']['humidity']
    wind_speed = weather_data['current']['wind_kph']
    weather_description = weather_data['current']['condition']['text']

    # Display weather information
    st.subheader(f"Weather in {CITY_NAME}")
    st.write(f"**Temperature**: {temperature}°C")
    st.write(f"**Humidity**: {humidity}%")
    st.write(f"**Wind Speed**: {wind_speed} km/h")
    st.write(f"**Description**: {weather_description}")

    # Display how weather can affect accidents
    st.write("""
    Weather conditions like rain, fog, snow, and high winds can significantly affect the likelihood of road accidents.
    It is important to adjust driving behavior according to weather conditions, such as reducing speed and increasing
    following distance during adverse weather.
    """)

    # Impact of the accident based on weather and user input
    impact = "Basic"  # Default impact

    # Determine severity based on weather conditions
    if 'rain' in weather_description.lower() or 'snow' in weather_description.lower() or 'fog' in weather_description.lower():
        impact = "Severe"
    elif 'cloudy' in weather_description.lower() or 'wind' in weather_description.lower():
        impact = "Intermediate"

    # Adjust impact based on user-selected conditions
    if road_condition in ['Wet', 'Slippery']:
        impact = "Severe"
    elif road_condition == 'Dry' and light_condition == 'Daylight':
        impact = "Basic"
    elif light_condition == 'Darkness':
        impact = "Intermediate"

    # Display the impact of the accident
    st.write(f"**Impact of the Accident**: {impact}")

    # **Plotting Graphs to Show Impact**
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.barplot(x=['Severe', 'Intermediate', 'Basic'], y=[1, 2, 3], ax=ax, palette='Blues')
    ax.set_title("Impact of Accident Based on Conditions")
    ax.set_ylabel("Accident Impact Score")
    st.pyplot(fig)

else:
    st.error("Failed to fetch weather data. Please try again later.")
