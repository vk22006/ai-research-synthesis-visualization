import streamlit as st

st.title("AI Research Knowledge Graph")

query = st.text_input("Enter research topic")

if st.button("Search"):
    st.write("Fetching papers...")