import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
import joblib

# Load the combined data
print("Loading traffic data...")
try:
    combinedData = pd.read_csv('combined_traffic_data1.csv')
    print(f"Data loaded successfully! Shape: {combinedData.shape}")
    print("\nColumns in dataset:")
    print(combinedData.columns.tolist())
except FileNotFoundError:
    print("Error: 'combined_traffic_data1.csv' not found!")
    exit()

# Display first few rows
print("\nFirst 5 rows of data:")
print(combinedData.head())

# --- 1. Create 4-way traffic model with 4 phases
print("\n--- Creating Traffic Phases ---")

# Convert Time column to datetime if it's not already
if 'Time' in combinedData.columns:
    try:
        combinedData['Time'] = pd.to_datetime(combinedData['Time'])
        time_seconds = (combinedData['Time'] - combinedData['Time'].iloc[0]).dt.total_seconds()
    except:
        # If conversion fails, create a sequential time series
        time_seconds = np.arange(len(combinedData))
        print("Warning: Could not parse Time column, using sequential values.")
else:
    time_seconds = np.arange(len(combinedData))
    print("Warning: No Time column found, using sequential values.")

# Define phase durations and cycle
phase_durations = [30, 5, 30, 5]  # NS Green, NS Yellow, EW Green, EW Yellow (seconds)
phase_cycle = ['NS_Green', 'NS_Yellow', 'EW_Green', 'EW_Yellow']
total_cycle_time = sum(phase_durations)

# Assign phases
def assign_phase(time_sec):
    cycle_position = time_sec % total_cycle_time
    cumulative_times = np.cumsum(phase_durations)
    phase_idx = np.searchsorted(cumulative_times, cycle_position)
    return phase_cycle[phase_idx]

combinedData['Phase'] = [assign_phase(t) for t in time_seconds]

# --- Add Direction based on Longitude and Latitude
print("\n--- Determining Vehicle Directions ---")

def determine_direction(lat, lon, ref_lat, ref_lon, tolerance=0.001):
    if lat > ref_lat and abs(lon - ref_lon) < tolerance:
        return 'North'
    elif lat < ref_lat and abs(lon - ref_lon) < tolerance:
        return 'South'
    elif lon > ref_lon and abs(lat - ref_lat) < tolerance:
        return 'East'
    elif lon < ref_lon and abs(lat - ref_lat) < tolerance:
        return 'West'
    else:
        return 'Unknown'

if 'Latitude' in combinedData.columns and 'Longitude' in combinedData.columns:
    ref_lat = combinedData['Latitude'].mean()
    ref_lon = combinedData['Longitude'].mean()
    combinedData['Direction'] = combinedData.apply(
        lambda row: determine_direction(row['Latitude'], row['Longitude'], ref_lat, ref_lon), 
        axis=1
    )
else:
    # Create dummy directions if coordinates not available
    combinedData['Direction'] = np.random.choice(['North', 'South', 'East', 'West'], len(combinedData))
    print("Warning: No coordinate columns found, using random directions.")

# --- 2. Prepare ground truth
print("\n--- Preparing Ground Truth Data ---")

# Create BrakeStatus if it doesn't exist
if 'BrakeStatus' not in combinedData.columns:
    print("Warning: BrakeStatus column not found. Creating dummy data.")
    combinedData['BrakeStatus'] = np.random.choice([0, 1], len(combinedData))

# Ensure Speed column exists
if 'Speed' not in combinedData.columns:
    print("Warning: Speed column not found. Creating dummy data.")
    combinedData['Speed'] = np.random.uniform(0, 80, len(combinedData))

# Identify hard braking events
combinedData['HardBraking'] = ((combinedData['Speed'] > 25) & (combinedData['BrakeStatus'] == 1)).astype(int)

# Display statistics
print(f"Total hard braking events: {combinedData['HardBraking'].sum()}")
print(f"Average speed: {combinedData['Speed'].mean():.2f}")

# --- 3. Create NEWS Score
print("\n--- Calculating NEWS Scores ---")

def calculate_news_score(speed, brake_status, hard_braking):
    score = 0
    if speed > 25:
        score += 3
    if brake_status == 1:
        score += 4
    if hard_braking == 1:
        score += 2
    return min(score, 9)

combinedData['NEWS_Score'] = combinedData.apply(
    lambda row: calculate_news_score(row['Speed'], row['BrakeStatus'], row['HardBraking']), 
    axis=1
)

# --- 4. Normalize Speed
print("\n--- Normalizing Speed Data ---")
speed_min = combinedData['Speed'].min()
speed_max = combinedData['Speed'].max()

if speed_max > speed_min:
    combinedData['Speed_Normalized'] = (combinedData['Speed'] - speed_min) / (speed_max - speed_min)
else:
    combinedData['Speed_Normalized'] = 0

# --- 5. Create visualizations
print("\n--- Creating Visualizations ---")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Figure 2: Direction Distribution
plt.figure(figsize=(10, 6))
direction_counts = combinedData['Direction'].value_counts()
plt.bar(direction_counts.index, direction_counts.values)
plt.title('Vehicle Distribution by Direction')
plt.xlabel('Direction')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('direction_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 6. Machine Learning Model
print("\n--- Training Machine Learning Model ---")

# Ensure all required columns exist, fill with zeros/defaults if missing
required_features = ['Speed', 'Heading', 'VehicleID', 'Longitude', 'Latitude', 'NEWS_Score', 'TimeRemaining', 'State', 'PositionalAccuracy']
for col in required_features:
    if col not in combinedData.columns:
        if col in ['Speed', 'Heading', 'Longitude', 'Latitude', 'NEWS_Score', 'TimeRemaining', 'PositionalAccuracy']:
            combinedData[col] = 0.0
        elif col in ['VehicleID']:
            combinedData[col] = 0
        elif col in ['State']:
            combinedData[col] = 'Unknown'

# Encode categorical columns
if 'State' in combinedData.columns:
    combinedData['State'] = LabelEncoder().fit_transform(combinedData['State'].astype(str))

# Prepare target
if 'BrakeStatus' not in combinedData.columns:
    print("Warning: BrakeStatus column not found. Creating dummy data.")
    combinedData['BrakeStatus'] = np.random.choice([0, 1], len(combinedData))

    y = combinedData['BrakeStatus']
X = combinedData[required_features].fillna(0)
    
# Train BaggingClassifier model (MATLAB fitcensemble Bag equivalent)
print("\n--- Training BaggingClassifier Model (fitcensemble Bag equivalent) ---")
    model = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=50,
        random_state=42
    )
    model.fit(X, y)
    
# Save the model
joblib.dump(model, 'bagging_model.pkl')
print("Model saved as bagging_model.pkl")

# Print features used
print("Features used for BaggingClassifier training:")
print(required_features)

# Feature importance plot (average over all trees)
feature_importance = np.zeros(len(required_features))
    for estimator in model.estimators_:
        feature_importance += estimator.feature_importances_
    feature_importance /= len(model.estimators_)
    indices = np.argsort(feature_importance)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importance[indices], align='center')
plt.xticks(range(X.shape[1]), [required_features[i] for i in indices], rotation=45)
plt.title('Feature Importance (BaggingClassifier)')
plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
print("\n--- Analysis Complete ---")

# --- 7. Save enhanced data
print("\n--- Saving Enhanced Dataset ---")

# Display column summary
print(f"\nFinal dataset shape: {combinedData.shape}")
print("New columns added:")
new_columns = ['Phase', 'Direction', 'HardBraking', 'NEWS_Score', 'Speed_Normalized', 
               'Phase_Encoded', 'Direction_Encoded']
for col in new_columns:
    if col in combinedData.columns:
        print(f"  - {col}")

# Save the enhanced dataset
combinedData.to_csv('enhanced_combined_traffic_data.csv', index=False)
print("\nEnhanced dataset saved as 'enhanced_combined_traffic_data.csv'")

# Save a focused dataset with key features
key_columns = ['Speed', 'BrakeStatus', 'Phase', 'Direction', 'HardBraking', 'NEWS_Score', 'Speed_Normalized']
key_columns = [col for col in key_columns if col in combinedData.columns]

if 'Time' in combinedData.columns:
    key_columns.insert(0, 'Time')
if 'VehicleID' in combinedData.columns:
    key_columns.insert(1, 'VehicleID')

focused_data = combinedData[key_columns]
focused_data.to_csv('new_features_only.csv', index=False)
print(f"Focused dataset saved as 'new_features_only.csv' with {len(key_columns)} columns")

print("\n--- Analysis Complete ---")
print("Generated files:")
print("  - enhanced_combined_traffic_data.csv")
print("  - new_features_only.csv")
print("  - phase_distribution.png")
print("  - direction_distribution.png") 
print("  - news_score_distribution.png")
print("  - speed_vs_news_score.png")
print("  - confusion_matrix.png")
print("  - feature_importance.png")

# Display final summary statistics
print(f"\nSummary Statistics:")
print(f"Total records: {len(combinedData)}")
print(f"Hard braking events: {combinedData['HardBraking'].sum()}")
print(f"Average NEWS Score: {combinedData['NEWS_Score'].mean():.2f}")
print(f"Unique phases: {combinedData['Phase'].nunique()}")
print(f"Unique directions: {combinedData['Direction'].nunique()}")