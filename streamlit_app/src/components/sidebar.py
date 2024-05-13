import streamlit as st

def show_sidebar():
    with st.sidebar:
        st.header('Sidebar Menu')
        
        st.write(
            """
                Here you need to specify your input parameters
            """
        )
