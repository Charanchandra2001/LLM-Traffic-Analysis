import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

# --- Configuration ---
DATA_FILE = 'combined_traffic_data1.csv'

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{DATA_FILE}' was not found. Please ensure it's in the same directory.")
        st.stop()

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
        avg_speed = filtered_df['Speed'].mean() if 'Speed' in filtered_df.columns else float('nan')
        st.metric(label="Average Speed", value=f"{avg_speed:.2f} km/h")

with col2:
    with st.container(border=True):
        avg_vehicle_length = filtered_df['VehicleLength'].mean() if 'VehicleLength' in filtered_df.columns else float('nan')
        st.metric(label="Average Vehicle Length", value=f"{avg_vehicle_length:.2f} m")

st.subheader("Traffic Speed Distribution")
if 'Speed' in filtered_df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_df['Speed'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Vehicle Speed')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No Speed data available for the selected filters.")

st.subheader("Brake Status Analysis")
if 'BrakeStatus' in filtered_df.columns:
    brake_status_counts = filtered_df['BrakeStatus'].value_counts().reset_index()
    brake_status_counts.columns = ['BrakeStatus', 'Count']
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='BrakeStatus', y='Count', data=brake_status_counts, ax=ax)
    ax.set_title('Counts of Brake Status Events')
    ax.set_xlabel('Brake Status')
    ax.set_ylabel('Number of Events')
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No BrakeStatus data available for the selected filters.")

# Add Vehicle Distribution by Direction visualization below the rest
st.subheader("Vehicle Distribution by Direction")
st.image('direction_distribution.png', caption='Vehicle Distribution by Direction', use_column_width=True)

# --- Dummy Event Data for Visualization ---
import pandas as pd
from datetime import datetime, timedelta

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
dummy_event_df['Minute'] = pd.to_datetime(dummy_event_df['Time']).dt.floor('T')

# --- Event Timeline and Map Visualization ---
st.subheader("Event Timeline (Hard Braking & Near Misses)")
# Timeline plot (events per minute)
event_counts_minute = dummy_event_df.groupby('Minute').size().reset_index(name='EventCount')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(event_counts_minute['Minute'], event_counts_minute['EventCount'], marker='o')
ax.set_title('Event Count per Minute')
ax.set_xlabel('Time (Minute)')
ax.set_ylabel('Event Count')
st.pyplot(fig)
plt.close(fig)

# Map visualization (if lat/lon available)
if 'Latitude' in dummy_event_df.columns and 'Longitude' in dummy_event_df.columns:
    st.subheader("Event Locations Map")
    map_df = dummy_event_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    st.map(map_df[['latitude', 'longitude']])
else:
    st.info("No latitude/longitude data available for event mapping.")

# Show event table
st.subheader("Event Details Table")
st.dataframe(dummy_event_df[['Time', 'VehicleID', 'EventType', 'Speed', 'SpeedDrop']].head(100))

# --- Additional Event Visualizations ---
st.subheader("Event Type Breakdown")
# Event type breakdown (bar and pie)
event_type_counts = dummy_event_df['EventType'].value_counts()
fig1, ax1 = plt.subplots()
event_type_counts.plot(kind='bar', ax=ax1)
ax1.set_title('Event Type Count')
ax1.set_xlabel('Event Type')
ax1.set_ylabel('Count')
st.pyplot(fig1)
plt.close(fig1)
fig2, ax2 = plt.subplots()
event_type_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
ax2.set_ylabel('')
ax2.set_title('Event Type Proportion')
st.pyplot(fig2)
plt.close(fig2)

# Event heatmap by hour of day
st.subheader("Event Heatmap by Hour of Day")
dummy_event_df['HourOfDay'] = pd.to_datetime(dummy_event_df['Time']).dt.hour
hour_counts = dummy_event_df.groupby('HourOfDay').size()
fig3, ax3 = plt.subplots()
sns.heatmap(hour_counts.values.reshape(1, -1), cmap='YlOrRd', annot=True, fmt='d', cbar=False, ax=ax3)
ax3.set_title('Events by Hour of Day')
ax3.set_xlabel('Hour of Day')
ax3.set_yticks([])
ax3.set_xticks(range(24))
ax3.set_xticklabels([str(h) for h in range(24)])
st.pyplot(fig3)
plt.close(fig3)

# Speed vs Event Type boxplot
st.subheader("Speed vs Event Type")
fig4, ax4 = plt.subplots()
sns.boxplot(x='EventType', y='Speed', data=dummy_event_df, ax=ax4)
ax4.set_title('Speed Distribution by Event Type')
st.pyplot(fig4)
plt.close(fig4)

# # Vehicle-specific event timeline
# st.subheader("Vehicle-specific Event Timeline")
# vehicle_ids = dummy_event_df['VehicleID'].unique()
# selected_vehicle = st.selectbox("Select Vehicle for Timeline", vehicle_ids)
# vdf = dummy_event_df[dummy_event_df['VehicleID'] == selected_vehicle]
# if not vdf.empty:
#     fig5, ax5 = plt.subplots(figsize=(10, 2))
#     ax5.scatter(vdf['Time'], [1]*len(vdf), c=vdf['EventType'].map({'HardBraking':'red','NearMiss':'blue'}), label='Event')
#     ax5.set_yticks([])
#     ax5.set_xlabel('Time')
#     ax5.set_title(f'Events for Vehicle {selected_vehicle}')
#     st.pyplot(fig5)
#     plt.close(fig5)
# else:
#     st.info("No events for selected vehicle.")
