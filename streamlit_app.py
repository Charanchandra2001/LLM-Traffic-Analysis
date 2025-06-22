import streamlit as st
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
DATA_FILE = 'combined_traffic_data1.csv'
TRAFFIC_REPORT_FILE = 'traffic_report.txt'

# Load OpenAI API Key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    st.error("OPENAI_API_KEY environment variable not set. Please set it before running the app.")
    st.stop()

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df

df = load_data()

# --- Load summary findings and stats from data.py output ---
def load_summary_findings():
    try:
        with open(TRAFFIC_REPORT_FILE, 'r') as f:
            summary = f.read()
        return summary
    except Exception:
        return ""

summary_findings = load_summary_findings()

# --- LLM Integration Setup ---
@st.cache_resource
def setup_llm_integration(dataframe, summary_findings):
    """Sets up the text data, splits it into chunks, and creates the FAISS vector store."""
    text_data = summary_findings + "\n\n" + dataframe.to_string(index=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text_data)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

vector_store = setup_llm_integration(df, summary_findings)

# --- Report Generation Function ---
def generate_report(query, vector_store_instance, openai_client_instance):
    """Generates a detailed report using the LLM based on relevant context."""
    docs = vector_store_instance.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
    You are a traffic analysis expert. Use the following context, which includes traffic data, summary findings, and statistics, to generate a detailed report or answer the query.
    Context: {context}
    Query: {query}
    Provide a structured report with sections: Summary, Analysis, Recommendations (if applicable), and thank the user for asking!
    """
    response = openai_client_instance.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.title("🚦 Traffic Analysis Report Generator")
st.markdown("Enter your query to get a detailed report on traffic conditions and safety insights, powered by an LLM.")

query = st.text_area(
    "**Enter your query here:**",
    "What are the traffic conditions, safety insights for high brake events, and the key findings and summary statistics from the latest data analysis?",
    height=100
)

if st.button("Generate Report", type="primary"):
    if query:
        with st.spinner("Generating report... This may take a moment."):
            report = generate_report(query, vector_store, openai_client)
            st.subheader("Generated Report")
            st.markdown(report)
    else:
        st.warning("Please enter a query to generate a report.")
