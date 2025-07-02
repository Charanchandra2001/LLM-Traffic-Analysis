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
from datetime import datetime, timedelta

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

def load_event_summary():
    try:
        df = pd.read_csv('event_summary.csv')
        return df.to_string(index=False)
    except Exception:
        return ""

event_summary_text = load_event_summary()

# --- Dummy Event Data for Analysis ---
dummy_now = datetime.now()
dummy_event_df = pd.DataFrame({
    'Time': [dummy_now, dummy_now + timedelta(minutes=1), dummy_now + timedelta(minutes=2)],
    'VehicleID': [1, 2, 1],
    'EventType': ['HardBraking', 'NearMiss', 'HardBraking'],
    'Speed': [30, 15, 28],
    'SpeedDrop': [12, 11, 13],
    'Latitude': [39.162, 39.163, 39.164],
    'Longitude': [-84.519, -84.520, -84.521]
})

def generate_dummy_event_analysis(df):
    # Event type breakdown
    event_type_counts = df['EventType'].value_counts()
    event_type_summary = ", ".join([f"{etype}: {count}" for etype, count in event_type_counts.items()])
    # Event timing
    times = pd.to_datetime(df['Time'])
    time_range = f"from {times.min().strftime('%Y-%m-%d %H:%M:%S')} to {times.max().strftime('%Y-%m-%d %H:%M:%S')}"
    # Speed distribution
    speed_stats = df.groupby('EventType')['Speed'].describe().to_dict()
    speed_summary = ""
    for etype in event_type_counts.index:
        stats = speed_stats['mean'][etype], speed_stats['min'][etype], speed_stats['max'][etype]
        speed_summary += f"{etype}: mean={stats[0]:.1f}, min={stats[1]}, max={stats[2]}. "
    # Vehicle-specific
    vehicle_events = df.groupby('VehicleID').size().to_dict()
    vehicle_summary = ", ".join([f"Vehicle {vid}: {count} events" for vid, count in vehicle_events.items()])
    # Compose analysis
    analysis = f"""
Dummy Event Data Analysis:
- Event types observed: {event_type_summary}
- Events occurred {time_range}.
- Speed by event type: {speed_summary}
- Events per vehicle: {vehicle_summary}
- Note: This is based on in-memory dummy data for demonstration purposes.
"""
    return analysis

dummy_event_analysis = generate_dummy_event_analysis(dummy_event_df)

# --- LLM Integration Setup ---
@st.cache_resource
def setup_llm_integration(dataframe, summary_findings, event_summary_text, dummy_event_analysis):
    """Sets up the text data, splits it into chunks, and creates the FAISS vector store."""
    text_data = summary_findings + "\n\n" + dummy_event_analysis + "\n\n" + event_summary_text + "\n\n" + dataframe.to_string(index=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text_data)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

vector_store = setup_llm_integration(df, summary_findings, event_summary_text, dummy_event_analysis)

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
    "****",
    "What are the traffic conditions, safety insights for high brake events, and the key findings and summary statistics from the latest data analysis?",
    height=100
)

if st.button("Generate Report", type="primary"):
    if query:
        with st.spinner("Generating report... This may take a moment."):
            report = generate_report(query, vector_store, openai_client)
            st.markdown(report)
    else:
        st.warning("Please enter a query to generate a report.")
