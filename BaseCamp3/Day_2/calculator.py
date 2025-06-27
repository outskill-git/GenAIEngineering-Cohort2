import streamlit as st
import requests
import json

st.title("Calculator App")
st.write("This app connects to a FastAPI calculator service.")

# Define the API base URL
<<<<<<< HEAD
=======
# api_url = "http://0.0.0.0:9321"
>>>>>>> efc449e (added week3 content)
api_url = "https://genaiengineering-cohort2-uzeu.onrender.com"

# Initialize session state to store the calculator display and current operation
if 'display' not in st.session_state:
    st.session_state.display = '0'
if 'first_number' not in st.session_state:
    st.session_state.first_number = None
if 'operation' not in st.session_state:
    st.session_state.operation = None
if 'expecting_second_number' not in st.session_state:
    st.session_state.expecting_second_number = False
if 'result' not in st.session_state:
    st.session_state.result = None
if 'api_response' not in st.session_state:
    st.session_state.api_response = None

# Display the calculator screen
st.text_input("Calculator Display", value=st.session_state.display, key="display_field", disabled=True)

# Function to handle number button clicks
def number_click(number):
    if st.session_state.expecting_second_number:
        st.session_state.display = str(number)
        st.session_state.expecting_second_number = False
    elif st.session_state.display == '0':
        st.session_state.display = str(number)
    else:
        st.session_state.display += str(number)

# Function to handle operation button clicks
def operation_click(op):
    st.session_state.first_number = float(st.session_state.display)
    st.session_state.operation = op
    st.session_state.expecting_second_number = True

# Function to clear the calculator
def clear_calculator():
    st.session_state.display = '0'
    st.session_state.first_number = None
    st.session_state.operation = None
    st.session_state.expecting_second_number = False
    st.session_state.result = None
    st.session_state.api_response = None

# Function to calculate result by calling the API
def calculate_result():
    try:
        if st.session_state.first_number is None or st.session_state.operation is None:
            return

        first_num = st.session_state.first_number
        second_num = float(st.session_state.display)

        # Construct the API request URL based on the selected operation
        endpoint = f"{api_url}/{st.session_state.operation}"

        # Make the API call
        # response = requests.get(endpoint, params={"a": first_num, "b": second_num})
        response = requests.post(endpoint, json={"a": first_num, "b": second_num})

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            st.session_state.result = result['result']
            st.session_state.api_response = result
            st.session_state.display = str(result['result'])
        else:
            st.session_state.display = f"Error: {response.status_code}"

    except requests.exceptions.ConnectionError:
        st.session_state.display = "Connection Error"
    except Exception as e:
        st.session_state.display = f"Error: {str(e)[:10]}"

# Create the calculator layout with CSS Grid-like appearance
col1, col2, col3, col4 = st.columns(4)

# Row 1 of the calculator (7, 8, 9, +)
with col1:
    st.button("7", on_click=number_click, args=(7,), use_container_width=True)
with col2:
    st.button("8", on_click=number_click, args=(8,), use_container_width=True)
with col3:
    st.button("9", on_click=number_click, args=(9,), use_container_width=True)
with col4:
    st.button("Add (+)", on_click=operation_click, args=("add",), use_container_width=True)

# Row 2 of the calculator (4, 5, 6, -)
with col1:
    st.button("4", on_click=number_click, args=(4,), use_container_width=True)
with col2:
    st.button("5", on_click=number_click, args=(5,), use_container_width=True)
with col3:
    st.button("6", on_click=number_click, args=(6,), use_container_width=True)
with col4:
    st.button("Sub (-)", on_click=operation_click, args=("subtract",), use_container_width=True)

# Row 3 of the calculator (1, 2, 3, C)
with col1:
    st.button("1", on_click=number_click, args=(1,), use_container_width=True)
with col2:
    st.button("2", on_click=number_click, args=(2,), use_container_width=True)
with col3:
    st.button("3", on_click=number_click, args=(3,), use_container_width=True)
with col4:
    st.button("C", on_click=clear_calculator, use_container_width=True)

# Row 4 of the calculator (0, ., =)
with col1:
    st.button("0", on_click=number_click, args=(0,), use_container_width=True)
with col2:
    st.button(".", on_click=lambda: setattr(st.session_state, 'display',
              st.session_state.display + '.' if '.' not in st.session_state.display else st.session_state.display),
              use_container_width=True)
with col3, col4:
    # Span the "=" button across two columns
    st.button("=", on_click=calculate_result, use_container_width=True)

# Display API response if available
if st.session_state.api_response:
    with st.expander("View API Response"):
        st.json(st.session_state.api_response)

# Add information about how to run the FastAPI server
st.markdown("---")
st.subheader("How to use this calculator")
st.markdown("""
1. Make sure the FastAPI calculator service is running at http://0.0.0.0:9321
2. Use the calculator buttons to input numbers and operations
3. Click "=" to calculate the result by calling the API
4. Click "C" to clear the calculator
""")

# Run with: streamlit run streamlit_calculator.py
