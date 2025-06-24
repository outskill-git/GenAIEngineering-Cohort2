import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

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
