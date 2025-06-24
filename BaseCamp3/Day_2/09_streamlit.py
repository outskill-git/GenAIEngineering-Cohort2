import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import io
import base64
import requests
import json
import time

# Basic Hello World app
st.title("Hello, Streamlit!")
st.write("Welcome to your first Streamlit app!")

# Adding different text elements
st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is plain text")
st.markdown("**This is markdown** with *formatting*")
st.caption("This is a small caption text")
st.code("print('Hello, Streamlit!')", language="python")

# Adding input widgets
st.header("Input Widgets")

# Text input
name = st.text_input("What's your name?")
if name:
    st.write(f"Hello, {name}!")

# Number input
age = st.number_input("How old are you?", min_value=0, max_value=120, step=1)
st.write(f"You are {age} years old.")

# Slider
height = st.slider("Select your height (in cm)", 100, 220, 170)
st.write(f"Your height is {height} cm.")

# Checkbox
agree = st.checkbox("I agree to the terms and conditions")
if agree:
    st.success("Thank you for agreeing!")

# Selectbox and buttons
st.header("Selectbox and Buttons")

# Selectbox
option = st.selectbox(
    'What is your favorite programming language?',
    ('Python', 'JavaScript', 'Java', 'C++', 'Other')
)
st.write(f"You selected: {option}")

# Radio buttons
level = st.radio(
    "What is your programming experience level?",
    ("Beginner", "Intermediate", "Advanced")
)
st.write(f"You are at the {level} level.")

# Buttons
if st.button("Say Hello"):
    st.write("Hello there!")
    
# Using columns for button layout
col1, col2 = st.columns(2)
with col1:
    if st.button("Show Success"):
        st.success("This is a success message!")
with col2:
    if st.button("Show Error"):
        st.error("This is an error message!")

# Layout containers
st.header("Layout Containers")

# Sidebar
st.sidebar.header("Sidebar")
st.sidebar.write("This is a sidebar where you can place widgets.")
sidebar_slider = st.sidebar.slider("Sidebar slider", 0, 100, 50)
st.sidebar.write(f"Slider value: {sidebar_slider}")

# Columns
st.subheader("Columns Layout")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Column 1")
    st.image("https://via.placeholder.com/150", caption="Placeholder Image")
with col2:
    st.write("Column 2")
    st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
with col3:
    st.write("Column 3")
    st.checkbox("Check me")

# Expander
with st.expander("Click to expand"):
    st.write("This content is initially hidden but can be expanded.")
    st.image("https://via.placeholder.com/300x100", caption="Wide Image")

# Container
with st.container():
    st.write("This is a container.")
    st.info("You can group elements together.")

# Data display
st.header("Data Display")

# Create a sample dataframe
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Age': [28, 34, 29, 42],
    'City': ['New York', 'Paris', 'Berlin', 'London'],
    'Salary': [65000, 70000, 62000, 85000]
}
df = pd.DataFrame(data)

# Display dataframe
st.subheader("DataFrame Display")
st.dataframe(df)

# Display as a static table
st.subheader("Static Table")
st.table(df)

# Display metrics
st.subheader("Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Average Age", f"{df['Age'].mean():.1f}", "years")
col2.metric("Total Employees", len(df), "+4")
col3.metric("Avg Salary", f"${df['Salary'].mean():.2f}", "$1,200")

# JSON display
st.subheader("JSON Display")
st.json({
    "name": "Streamlit",
    "version": "1.20.0",
    "features": ["Easy to use", "Fast prototyping", "Interactive widgets"]
})

# Code display
st.subheader("Code Display")
code = '''
def hello_world():
    print("Hello, Streamlit!")
    return "Success"
'''
st.code(code, language="python")

# Charts and Visualizations
st.header("Charts and Visualizations")

# Create sample data for charts
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)

# Line chart
st.subheader("Line Chart")
st.line_chart(chart_data)

# Area chart
st.subheader("Area Chart")
st.area_chart(chart_data)

# Bar chart
st.subheader("Bar Chart")
st.bar_chart(chart_data)

# Matplotlib integration
st.subheader("Matplotlib Chart")
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(chart_data.index, chart_data['A'], label='A', color='blue', alpha=0.7)
ax.scatter(chart_data.index, chart_data['B'], label='B', color='red', alpha=0.7)
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.set_title('Scatter Plot with Matplotlib')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Altair chart
st.subheader("Altair Chart")
chart = alt.Chart(chart_data.reset_index()).mark_circle().encode(
    x='index',
    y='A',
    size='B',
    color='C',
    tooltip=['index', 'A', 'B', 'C']
).interactive()
st.altair_chart(chart, use_container_width=True)

# Map
st.subheader("Map Display")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)
st.map(map_data)

# File upload and download
st.header("File Upload and Download")

# File uploader
st.subheader("File Upload")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Read the file
    df_upload = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df_upload.head())
    
    # Show basic statistics
    st.write("Data Statistics:")
    st.write(df_upload.describe())
    
    # Plot uploaded data
    st.write("Data Visualization:")
    if st.checkbox("Show Plot for Uploaded Data"):
        try:
            # Try to create a chart from the uploaded data
            st.line_chart(df_upload.select_dtypes(include=['float64', 'int64']).iloc[:, :3])
        except Exception as e:
            st.error(f"Could not plot data: {e}")

# File download
st.subheader("File Download")

# Create a sample dataframe for download
download_data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Value 1': [10, 20, 30, 40],
    'Value 2': [100, 200, 300, 400]
})

# CSV download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(download_data)
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name='sample_data.csv',
    mime='text/csv',
)

# Excel download
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    processed_data = output.getvalue()
    return processed_data

excel = convert_df_to_excel(download_data)

# API Requests and Data Display
st.header("API Requests and Server Data")

# Public API demonstration
st.subheader("Fetching Data from Public APIs")

# Function to fetch data from an API
@st.cache_data(ttl=600)  # Cache the data for 10 minutes
def fetch_api_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
        return None

# Example 1: Fetching random user data
if st.button("Fetch Random User Data"):
    with st.spinner("Fetching user data..."):
        data = fetch_api_data("https://randomuser.me/api/")
        if data:
            user = data['results'][0]
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(user['picture']['large'], width=150)
            with col2:
                st.write(f"**Name:** {user['name']['first']} {user['name']['last']}")
                st.write(f"**Email:** {user['email']}")
                st.write(f"**Location:** {user['location']['city']}, {user['location']['country']}")
                st.write(f"**Phone:** {user['phone']}")
            
            # Display raw JSON
            if st.checkbox("Show raw JSON"):
                st.json(user)

# Example 2: Fetching cryptocurrency prices
crypto_options = ['bitcoin', 'ethereum', 'litecoin', 'dogecoin', 'ripple']
selected_crypto = st.selectbox("Select a cryptocurrency", crypto_options)

if st.button("Get Crypto Price"):
    crypto_url = f"https://api.coingecko.com/api/v3/simple/price?ids={selected_crypto}&vs_currencies=usd,eur,gbp"
    with st.spinner(f"Fetching {selected_crypto} price..."):
        price_data = fetch_api_data(crypto_url)
        if price_data:
            st.success(f"Current {selected_crypto.title()} Prices:")
            col1, col2, col3 = st.columns(3)
            col1.metric("USD", f"${price_data[selected_crypto]['usd']:,.2f}")
            col2.metric("EUR", f"€{price_data[selected_crypto]['eur']:,.2f}")
            col3.metric("GBP", f"£{price_data[selected_crypto]['gbp']:,.2f}")

# Example 3: Weather data
st.subheader("Weather Data API")
city = st.text_input("Enter a city name:", "London")

if st.button("Get Weather"):
    # Note: In a real application, you would use your own API key
    # For demonstration purposes, we'll create mock data
    with st.spinner(f"Fetching weather data for {city}..."):
        # Simulate API call delay
        time.sleep(1)
        
        # Generate mock weather data
        weather_data = {
            "city": city,
            "temperature": round(np.random.uniform(5, 30), 1),
            "humidity": round(np.random.uniform(30, 90)),
            "wind_speed": round(np.random.uniform(0, 30), 1),
            "conditions": np.random.choice(["Sunny", "Cloudy", "Rainy", "Snowy", "Partly Cloudy"]),
            "forecast": [
                {"day": "Today", "high": round(np.random.uniform(15, 35)), "low": round(np.random.uniform(5, 15))},
                {"day": "Tomorrow", "high": round(np.random.uniform(15, 35)), "low": round(np.random.uniform(5, 15))},
                {"day": "Day After", "high": round(np.random.uniform(15, 35)), "low": round(np.random.uniform(5, 15))}
            ]
        }
        
        # Display current weather
        st.write(f"### Current Weather in {weather_data['city']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{weather_data['temperature']}°C")
        col2.metric("Humidity", f"{weather_data['humidity']}%")
        col3.metric("Wind Speed", f"{weather_data['wind_speed']} km/h")
        
        st.info(f"Conditions: {weather_data['conditions']}")
        
        # Display forecast
        st.write("### 3-Day Forecast")
        forecast_df = pd.DataFrame(weather_data['forecast'])
        st.table(forecast_df)
        
        # Create a chart for the forecast
        fig, ax = plt.subplots(figsize=(10, 4))
        days = [day["day"] for day in weather_data['forecast']]
        highs = [day["high"] for day in weather_data['forecast']]
        lows = [day["low"] for day in weather_data['forecast']]
        
        ax.bar(days, highs, label='High', alpha=0.7, color='red')
        ax.bar(days, lows, label='Low', alpha=0.7, color='blue')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Forecast')
        ax.legend()
        
        st.pyplot(fig)

# Example 4: Custom API endpoint with authentication
st.subheader("Custom API with Authentication")

# Input fields for API configuration
api_url = st.text_input("API Endpoint URL", "https://api.example.com/data")
use_auth = st.checkbox("Use Authentication")

auth_token = None
if use_auth:
    auth_type = st.radio("Authentication Type", ["API Key", "Bearer Token", "Basic Auth"])
    
    if auth_type == "API Key":
        api_key = st.text_input("API Key", type="password")
        key_name = st.text_input("Key Parameter Name", "api_key")
        auth_token = {"type": "api_key", "key": api_key, "name": key_name}
    
    elif auth_type == "Bearer Token":
        bearer_token = st.text_input("Bearer Token", type="password")
        auth_token = {"type": "bearer", "token": bearer_token}
    
    elif auth_type == "Basic Auth":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        auth_token = {"type": "basic", "username": username, "password": password}

if st.button("Send API Request"):
    with st.spinner("Sending request to API..."):
        try:
            # In a real application, you would use the auth_token to configure your request
            # For demonstration, we'll create mock response data
            
            # Simulate API call delay
            time.sleep(1.5)
            
            # Mock successful API response
            mock_response = {
                "status": "success",
                "data": {
                    "items": [
                        {"id": 1, "name": "Item 1", "value": round(np.random.uniform(100, 1000), 2)},
                        {"id": 2, "name": "Item 2", "value": round(np.random.uniform(100, 1000), 2)},
                        {"id": 3, "name": "Item 3", "value": round(np.random.uniform(100, 1000), 2)},
                        {"id": 4, "name": "Item 4", "value": round(np.random.uniform(100, 1000), 2)},
                        {"id": 5, "name": "Item 5", "value": round(np.random.uniform(100, 1000), 2)}
                    ],
                    "total": 5,
                    "page": 1
                }
            }
            
            # Display the response
            st.success("Request successful!")
            
            # Convert to DataFrame for display
            result_df = pd.DataFrame(mock_response["data"]["items"])
            st.dataframe(result_df)
            
            # Create a chart from the data
            st.subheader("Data Visualization")
            chart = alt.Chart(result_df).mark_bar().encode(
                x='name',
                y='value',
                color='name'
            ).properties(width=600)
            st.altair_chart(chart, use_container_width=True)
            
            # Show raw JSON
            if st.checkbox("Show Raw Response"):
                st.json(mock_response)
                
        except Exception as e:
            st.error(f"Error making API request: {e}")

# Connection status monitor
st.subheader("API Connection Status")

# Mock service status
services = [
    {"name": "Database Server", "status": "Online", "latency": "25ms"},
    {"name": "Auth Service", "status": "Online", "latency": "48ms"},
    {"name": "Storage Service", "status": "Online", "latency": "63ms"},
    {"name": "Analytics API", "status": "Degraded", "latency": "187ms"},
    {"name": "Email Service", "status": "Offline", "latency": "N/A"}
]

# Display service status
st.write("Current Service Status:")
status_df = pd.DataFrame(services)

# Color-code the status
def highlight_status(s):
    if s == 'Online':
        return 'background-color: #a5d6a7; color: black'
    elif s == 'Degraded':
        return 'background-color: #fff59d; color: black'
    elif s == 'Offline':
        return 'background-color: #ef9a9a; color: black'
    return ''

# Display styled dataframe
st.dataframe(status_df.style.applymap(highlight_status, subset=['status']))

# Auto-refresh option
if st.checkbox("Enable auto-refresh (every 60 seconds)"):
    st.info("Status will automatically refresh every 60 seconds. (Note: in a real app, this would use a session state to track time.)")
    current_time = time.strftime("%H:%M:%S", time.localtime())
    st.write(f"Last updated: {current_time}")
