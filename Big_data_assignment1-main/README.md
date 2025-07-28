# Big_data_assignment1
# Uber Fare Data Analysis Project
### 🔍 Course: INSY 8413 - Introduction to Big Data Analytics  
*Instructor:* Eric Maniraguha  
*Student:* Nkuranga James(26285) 
*Date:* July 2025

## Overview
This project analyzes Uber ride data to understand fare patterns, identify outliers, and create meaningful visualizations. The analysis includes data cleaning, feature engineering, statistical analysis, and comprehensive visualizations.

## Dataset Information
- *Original Dataset*: uber.csv
- *Cleaned Dataset*: uber_enhanced.csv
- *Initial Shape*: 200,000 rows × 9 columns
- *Final Shape*: Reduced after removing missing values and unrealistic distances

## Data Cleaning Process

### 1. Initial Data Exploration
```python
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('uber.csv')

# Dataset structure and summary
print("Initial Shape:", df.shape)
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())
```

### 2. Data Cleaning Steps
```python
# Drop rows with missing values
df_clean = df.dropna()
print("Shape after dropping missing values:", df_clean.shape)

# Descriptive statistics
print("Descriptive statistics:\n", df_clean.describe())
print("Mode:\n", df_clean.mode(numeric_only=True))

# Calculate fare range
fare_range = df_clean['fare_amount'].max() - df_clean['fare_amount'].min()
print("Fare Range:", fare_range)
```

## Outlier Detection

### IQR Method for Fare Amount
```python
# Outlier detection using IQR method
Q1 = df_clean['fare_amount'].quantile(0.25)
Q3 = df_clean['fare_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['fare_amount'] < lower_bound) | (df_clean['fare_amount'] > upper_bound)]
print("Number of outliers in fare_amount:", len(outliers))
```

## Feature Engineering

### DateTime Feature Extraction
```python
# Convert pickup datetime to datetime format
df_clean['pickup_datetime'] = pd.to_datetime(df_clean['pickup_datetime'])

# Create new features
df_clean['hour'] = df_clean['pickup_datetime'].dt.hour
df_clean['day'] = df_clean['pickup_datetime'].dt.day
df_clean['month'] = df_clean['pickup_datetime'].dt.month
df_clean['weekday'] = df_clean['pickup_datetime'].dt.dayofweek
df_clean['period'] = df_clean['hour'].apply(lambda x: 'Peak' if 7 <= x <= 9 or 17 <= x <= 20 else 'Off-Peak')
```

### Distance Calculation using Haversine Formula
```python
def haversine(row):
    """Calculate the great circle distance between two points on Earth"""
    lat1, lon1, lat2, lon2 = map(radians, [
        row['pickup_latitude'], row['pickup_longitude'], 
        row['dropoff_latitude'], row['dropoff_longitude']
    ])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6371 * c
    return km

# Calculate distance and add as new column
df_clean['distance_km'] = df_clean.apply(haversine, axis=1)

# Filter out unrealistic distances
df_clean = df_clean[(df_clean['distance_km'] > 0) & (df_clean['distance_km'] < 100)]
```

## Visualizations Generated

### 1. Fare Distribution Analysis
```python
# Fare distribution histogram
sns.histplot(df_clean['fare_amount'], bins=50)
plt.title('Fare Amount Distribution')
plt.savefig('fare_distribution.png')
plt.clf()
```
<img width="640" height="480" alt="fare_distribution" src="https://github.com/user-attachments/assets/6c9127bc-dfa9-4819-b92a-4c6d50d13486" />


*Insights*: The fare distribution shows a right-skewed pattern with most fares concentrated in the lower range (0-50), indicating that most Uber rides are relatively short-distance trips.

### 2. Outlier Detection Visualization
```python
# Box plot for outlier detection
sns.boxplot(x=df_clean['fare_amount'])
plt.title('Fare Amount Outliers')
plt.savefig('fare_boxplot.png')
plt.clf()
```

<img width="640" height="480" alt="fare_boxplot" src="https://github.com/user-attachments/assets/8d917919-4201-4b98-9362-89cebd232485" />

*Insights*: The box plot reveals significant outliers in fare amounts, with some fares extending well beyond the typical range, suggesting either very long trips or data anomalies.

### 3. Fare vs Distance Analysis
```python
# Scatter plot of fare vs distance
sns.scatterplot(x='distance_km', y='fare_amount', data=df_clean)
plt.title("Fare Amount vs Distance (km)")
plt.xlabel("Distance (km)")
plt.ylabel("Fare Amount ($)")
plt.savefig('fare_vs_distance.png')
plt.clf()
```
<img width="640" height="480" alt="fare_vs_distance" src="https://github.com/user-attachments/assets/36442c16-a971-455b-8f52-6a0bfa5549ae" />


*Insights*: Shows a positive correlation between distance and fare amount, with most trips concentrated in the 0-20km range. Some outliers suggest premium pricing or surge pricing scenarios.

### 4. Temporal Analysis - Fare by Hour
```python
# Box plot of fare by hour of day
sns.boxplot(x='hour', y='fare_amount', data=df_clean)
plt.title('Fare Amount by Hour of Day')
plt.savefig('fare_by_hour.png')
plt.clf()
```
<img width="640" height="480" alt="fare_by_hour" src="https://github.com/user-attachments/assets/a6b6a989-38cb-432a-b26a-cb5b25759f8d" />


*Insights*: Fare patterns vary throughout the day, with potential surge pricing during peak hours. The visualization helps identify high-demand periods.

### 5. Correlation Analysis
```python
# Correlation heatmap
corr = df_clean[['fare_amount', 'hour', 'day', 'month', 'weekday', 'distance_km']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.clf()
```
<img width="640" height="480" alt="correlation_matrix" src="https://github.com/user-attachments/assets/bd454bd1-f618-40dc-ac7f-cda29526fea1" />


*Insights*: The correlation matrix reveals that distance_km has the strongest positive correlation (0.84) with fare_amount, which aligns with expectations. Other temporal features show weaker correlations.

## Data Export
```python
# Export cleaned and enhanced dataset
df_clean.to_csv('uber_enhanced.csv', index=False)
## OUTPUTS 
<img width="635" height="488" alt="Screenshot 2025-07-25 180016" src="https://github.com/user-attachments/assets/235ac20b-18f2-45cc-890a-24f34efb96e8" />
<img width="652" height="510" alt="Screenshot 2025-07-25 180109" src="https://github.com/user-attachments/assets/1474b215-7e6a-483b-b822-4197a7c701be" />
```
## Outputs
<img width="635" height="488" alt="Screenshot 2025-07-25 180016" src="https://github.com/user-attachments/assets/51480bca-5d2f-4211-a731-af4b441a7d5d" />

<img width="652" height="510" alt="Screenshot 2025-07-25 180109" src="https://github.com/user-attachments/assets/eb1183e9-964a-4ab0-a162-6ea488702a0a" />

## Dashboard
<img width="960" height="538" alt="Dasboard" src="https://github.com/user-attachments/assets/7ebb97f5-5d34-4d38-82f3-86b2e53b6b79" />



## Key Findings

1. *Strong Distance-Fare Relationship*: Distance is the primary factor determining fare amount (correlation: 0.84)
2. *Outlier Presence*: Significant outliers in fare amounts require investigation
3. *Temporal Patterns*: Fare variations exist across different hours of the day
4. *Data Quality*: Original dataset had missing values that were successfully cleaned
5. *Feature Engineering*: Successfully created meaningful time-based and distance features

## Enhanced Dataset Features

The cleaned dataset (uber_enhanced.csv) includes:
- Original features: fare_amount, pickup/dropoff coordinates, datetime, passenger_count
- New features: hour, day, month, weekday, period (Peak/Off-Peak), distance_km

## Usage Instructions

1. Ensure you have the required libraries installed:
   bash
   pip install pandas matplotlib seaborn
   

2. Place your uber.csv file in the same directory as the script

3. Run the analysis script to generate all visualizations and the cleaned dataset

4. Review the generated PNG files for insights and the uber_enhanced.csv for further analysis

## Business Implications

- *Pricing Strategy*: Distance-based pricing is well-established
- *Demand Patterns*: Temporal analysis can inform surge pricing strategies  
- *Data Quality*: Regular data cleaning processes are essential for accurate analysis
- *Outlier Investigation*: High-fare outliers may indicate premium services or data errors

---

This analysis provides a foundation for understanding Uber fare patterns and can be extended with additional features like weather data, traffic conditions, or event-based demand spikes.
