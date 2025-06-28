import streamlit as st
import pandas as pd
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="PDF Chunk Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI service URL
API_URL = "http://localhost:9321/search_chunks"  # Update this with your actual API URL

def search_pdf_chunks(search_string):
    payload={'search_string': search_string}
    print(API_URL, payload)
    response = requests.post(f"{API_URL}?search_string={search_string}")
    if response.status_code == 200:
        return pd.read_json(response.json())
    else:
        st.error(f"Error: API returned status code {response.status_code}")
        st.error(response.text)
        return None

def main():
    # App title and description
    st.title("📄 PDF Chunk Search")
    st.markdown("""
    Search through all PDF chunks for specific keywords or phrases.
    Results will show matching chunks with their source document and page number.
    """)

    # Search input
    col1, col2 = st.columns([3, 1])

    with col1:
        search_input = st.text_input("Search for keywords or phrases:",
                                     placeholder="Enter your search terms...")

    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Display search results
    if search_button and search_input:
        st.subheader("Search Results")

        # Show a spinner while waiting for results
        with st.spinner(f"Searching for '{search_input}'..."):
            # Call the API
            results = search_pdf_chunks(search_input)
        # Display results
        if results is not None:
            if len(results) > 0:
                st.success(f"Found {len(results)} matching chunks")

                # Add a filter for filename
                if len(results['filename'].unique()) > 1:
                    file_filter = st.multiselect(
                        "Filter by document:",
                        options=sorted(results['filename'].unique()),
                        default=sorted(results['filename'].unique())
                    )

                    # Apply file filter
                    if file_filter:
                        results = results[results['filename'].isin(file_filter)]

                # Display results in an expander for each chunk
                for index, row in results.iterrows():
                    with st.expander(f"📄 {row['filename']} - Page {row['page_number']} - Chunk {row['chunk_number']}", expanded=True):
                        # Create a bordered box for the chunk text
                        st.markdown("""
                        <style>
                        .chunk-box {
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            padding: 10px;
                            background-color: #f9f9f9;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        # Display chunk text
                        st.markdown(f"<div class='chunk-box'>{row['chunk']}</div>", unsafe_allow_html=True)

                        # Add metadata at the bottom
                        st.caption(f"Document: {row['filename']} | Page: {row['page_number']} | Chunk: {row['chunk_number']}")
            else:
                st.warning(f"No results found for '{search_input}'")

    # Show instructions when no search has been performed
    if not search_button or not search_input:
        st.info("Enter a search term and click 'Search' to find matching chunks from processed PDFs.")

        # Add some example searches
        st.markdown("### Example searches:")
        example_searches = ["important", "data", "analysis", "conclusion"]

        for example in example_searches:
            if st.button(example):
                search_input = example
                # Re-run the app with the new input
                st.experimental_rerun()

if __name__ == "__main__":
    main()