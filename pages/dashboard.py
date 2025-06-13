# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# DATA_FILE = 'combined_traffic_data.csv'

# @st.cache_data
# def load_data():
#     return pd.read_csv(DATA_FILE)

# df = load_data()

# st.sidebar.header("Dashboard Filters")

# vehicle_ids = df['VehicleID'].unique()
# selected_vehicle_id = st.sidebar.selectbox("Select Vehicle ID", ['All'] + list(vehicle_ids))

# filtered_df = df.copy()
# if selected_vehicle_id != 'All':
#     filtered_df = df[df['VehicleID'] == selected_vehicle_id]

# # Display key metrics at the top
# st.subheader("Key Traffic Metrics")

# col1, col2 = st.columns(2)

# with col1:
#     with st.container(border=True):
#         avg_speed = filtered_df['Speed'].mean()
#         st.metric(label="Average Speed", value=f"{avg_speed:.2f} km/h")

# with col2:
#     with st.container(border=True):
#         avg_vehicle_length = filtered_df['VehicleLength'].mean()
#         st.metric(label="Average Vehicle Length", value=f"{avg_vehicle_length:.2f} m")

# st.subheader("Traffic Speed Distribution")
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.histplot(filtered_df['Speed'], bins=30, kde=True, ax=ax)
# ax.set_title('Distribution of Vehicle Speed')
# ax.set_xlabel('Speed')
# ax.set_ylabel('Frequency')
# st.pyplot(fig)
# plt.close(fig)

# # st.subheader("Brake Status Analysis")
# # brake_status_counts = filtered_df['BrakeStatus'].value_counts().reset_index()
# # brake_status_counts.columns = ['BrakeStatus', 'Count']
# # fig, ax = plt.subplots(figsize=(8, 6))
# # sns.barplot(x='BrakeStatus', y='Count', data=brake_status_counts, ax=ax)
# # ax.set_title('Counts of Brake Status Events')
# # ax.set_xlabel('Brake Status')
# # ax.set_ylabel('Number of Events')
# # st.pyplot(fig)
# # plt.close(fig) 

# st.subheader("Brake Status Analysis")
# brake_status_counts = filtered_df['BrakeStatus'].value_counts().reset_index()
# brake_status_counts.columns = ['BrakeStatus', 'Count']
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.barplot(x='BrakeStatus', y='Count', data=brake_status_counts, ax=ax, **hue='BrakeStatus'**)
# ax.set_title('Counts of Brake Status Events')
# ax.set_xlabel('Brake Status')
# ax.set_ylabel('Number of Events')
# st.pyplot(fig)
# plt.close(fig)



import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # Import numpy for demonstration

# --- Configuration ---
DATA_FILE = 'combined_traffic_data.csv' # Make sure this file exists in your project directory

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads the traffic data from a CSV file."""
    try:
        df = pd.read_csv(DATA_FILE)
        # Ensure 'BrakeStatus' column exists for the new plot
        if 'BrakeStatus' not in df.columns:
            # Add dummy BrakeStatus for demonstration if it's missing
            # In a real scenario, you'd ensure your data has this column
            np.random.seed(42)
            df['BrakeStatus'] = np.random.choice(['Normal', 'Warning', 'Fault'], size=len(df))
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{DATA_FILE}' was not found. Please ensure it's in the same directory.")
        st.stop() # Stop the app if data file is missing

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")

vehicle_ids = df['VehicleID'].unique()
selected_vehicle_id = st.sidebar.selectbox("Select Vehicle ID", ['All'] + list(vehicle_ids))

# Apply filter
filtered_df = df.copy()
if selected_vehicle_id != 'All':
    filtered_df = df[df['VehicleID'] == selected_vehicle_id]

# --- Key Metrics Display ---
st.subheader("Key Traffic Metrics")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        if not filtered_df.empty:
            avg_speed = filtered_df['Speed'].mean()
            st.metric(label="Average Speed", value=f"{avg_speed:.2f} km/h")
        else:
            st.metric(label="Average Speed", value="N/A")

with col2:
    with st.container(border=True):
        if not filtered_df.empty:
            avg_vehicle_length = filtered_df['VehicleLength'].mean()
            st.metric(label="Average Vehicle Length", value=f"{avg_vehicle_length:.2f} m")
        else:
            st.metric(label="Average Vehicle Length", value="N/A")

# --- Traffic Speed Distribution Plot ---
st.subheader("Traffic Speed Distribution")
if not filtered_df.empty:
    fig_speed, ax_speed = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_df['Speed'], bins=30, kde=True, ax=ax_speed)
    ax_speed.set_title('Distribution of Vehicle Speed')
    ax_speed.set_xlabel('Speed')
    ax_speed.set_ylabel('Frequency')
    st.pyplot(fig_speed)
    plt.close(fig_speed)
else:
    st.info("No data available for the selected filters to display speed distribution.")

# --- Brake Status Analysis Plot ---
st.subheader("Brake Status Analysis")
if not filtered_df.empty and 'BrakeStatus' in filtered_df.columns:
    brake_status_counts = filtered_df['BrakeStatus'].value_counts().reset_index()
    brake_status_counts.columns = ['BrakeStatus', 'Count']

    fig_brake, ax_brake = plt.subplots(figsize=(8, 6))
    # Use 'hue' to assign different colors based on 'BrakeStatus'
    sns.barplot(x='BrakeStatus', y='Count', data=brake_status_counts, ax=ax_brake, hue='BrakeStatus', palette='viridis', legend=False)

    ax_brake.set_title('Counts of Brake Status Events')
    ax_brake.set_xlabel('Brake Status')
    ax_brake.set_ylabel('Number of Events')
    st.pyplot(fig_brake)
    plt.close(fig_brake)
elif 'BrakeStatus' not in filtered_df.columns:
    st.warning("The 'BrakeStatus' column is not found in your data. Please ensure your 'combined_traffic_data.csv' file contains this column for the Brake Status Analysis.")
else:
    st.info("No data available for the selected filters to display brake status.")
