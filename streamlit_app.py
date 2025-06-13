import streamlit as st
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import tiktoken

# Environment and API setup
from dotenv import load_dotenv
import os
from openai import OpenAI

# LangChain and vector store setup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Ensure your 'combined_traffic_data.csv' and 'xgboost_model.pkl' are in the same directory as this script,
# or provide their full paths.
DATA_FILE = 'combined_traffic_data.csv'
MODEL_FILE = 'xgboost_model.pkl'

# Load OpenAI API Key from environment variables
# It's recommended to set this as an environment variable (e.g., OPENAI_API_KEY=your_key_here)
# before running the Streamlit app.
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    st.error("OPENAI_API_KEY environment variable not set. Please set it before running the app.")
    st.stop()

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

# --- Data Loading and Model Setup ---
@st.cache_data
def load_data_and_model():
    """Loads the traffic data and the trained XGBoost model."""
    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    return df, model

df, model = load_data_and_model()

# --- Model Performance Metrics (placeholder for display) ---
# In a real-world scenario, you would typically load actual evaluation metrics
# saved during your model training phase or recalculate them from a held-out test set.
# For this Streamlit app demonstration, we'll use placeholder values.
metrics = {
    "Accuracy": 1.0,
    "Precision": 0.98,
    "Recall": 0.95,
    "F1-Score": 0.98
}

# --- LLM Integration Setup ---
@st.cache_resource
def setup_llm_integration(dataframe, evaluation_metrics):
    """Sets up the text data, splits it into chunks, and creates the FAISS vector store."""
    metrics_text = "\nModel Performance Metrics:\n" + "\n".join([f"{k}: {v}" for k, v in evaluation_metrics.items()])
    text_data = dataframe.to_string(index=False) + "\n\n" + metrics_text

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text_data)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

vector_store = setup_llm_integration(df, metrics)

# --- Report Generation Function ---
def generate_report(query, vector_store_instance, openai_client_instance):
    """Generates a detailed report using the LLM based on relevant context."""
    docs = vector_store_instance.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a traffic analysis expert. Use the following context, which includes traffic data and model performance metrics, to generate a detailed report or answer the query.
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
    "What are the traffic conditions and safety insights for high brake events, including model performance?",
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
