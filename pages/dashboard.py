import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_FILE = 'combined_traffic_data.csv'

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

st.sidebar.header("Dashboard Filters")

vehicle_ids = df['VehicleID'].unique()
selected_vehicle_id = st.sidebar.selectbox("Select Vehicle ID", ['All'] + list(vehicle_ids))

filtered_df = df.copy()
if selected_vehicle_id != 'All':
    filtered_df = df[df['VehicleID'] == selected_vehicle_id]

# Display key metrics at the top
st.subheader("Key Traffic Metrics")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        avg_speed = filtered_df['Speed'].mean()
        st.metric(label="Average Speed", value=f"{avg_speed:.2f} km/h")

with col2:
    with st.container(border=True):
        avg_vehicle_length = filtered_df['VehicleLength'].mean()
        st.metric(label="Average Vehicle Length", value=f"{avg_vehicle_length:.2f} m")

st.subheader("Traffic Speed Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df['Speed'], bins=30, kde=True, ax=ax)
ax.set_title('Distribution of Vehicle Speed')
ax.set_xlabel('Speed')
ax.set_ylabel('Frequency')
st.pyplot(fig)
plt.close(fig)

# st.subheader("Brake Status Analysis")
# brake_status_counts = filtered_df['BrakeStatus'].value_counts().reset_index()
# brake_status_counts.columns = ['BrakeStatus', 'Count']
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.barplot(x='BrakeStatus', y='Count', data=brake_status_counts, ax=ax)
# ax.set_title('Counts of Brake Status Events')
# ax.set_xlabel('Brake Status')
# ax.set_ylabel('Number of Events')
# st.pyplot(fig)
# plt.close(fig) 

st.subheader("Brake Status Analysis")
brake_status_counts = filtered_df['BrakeStatus'].value_counts().reset_index()
brake_status_counts.columns = ['BrakeStatus', 'Count']
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='BrakeStatus', y='Count', data=brake_status_counts, ax=ax, **hue='BrakeStatus'**)
ax.set_title('Counts of Brake Status Events')
ax.set_xlabel('Brake Status')
ax.set_ylabel('Number of Events')
st.pyplot(fig)
plt.close(fig)
