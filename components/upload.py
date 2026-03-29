import streamlit as st
import requests
from utils.api import upload_pdfs_api

def render_uploader():
    st.sidebar.header("Upload Medical documnets (.PDFs)")
    uploaded_files=st.sidebar.file_uploader("Upload multiple PDFs",accept_multiple_files=True)
    if st.sidebar.button("Upload DB") and uploaded_files:
        response=upload_pdfs_api
        if response.status_code==200:
            st.sidebar.success("Uploaded Successfully")
        else:
            st.sidebar.error(f"Error:{response.text}")

