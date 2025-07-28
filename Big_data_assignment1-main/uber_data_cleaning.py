import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('uber.csv')  # Replace with your actual file path

# Dataset structure and summary
print("Initial Shape:", df.shape)
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())

# Drop rows with missing values
df_clean = df.dropna()
print("Shape after dropping missing values:", df_clean.shape)

# Descriptive statistics
print("Descriptive statistics:\n", df_clean.describe())

# Mode calculation
print("Mode:\n", df_clean.mode(numeric_only=True))

# Calculate fare range
fare_range = df_clean['fare_amount'].max() - df_clean['fare_amount'].min()
print("Fare Range:", fare_range)

# Outlier detection using IQR method for fare_amount
Q1 = df_clean['fare_amount'].quantile(0.25)
Q3 = df_clean['fare_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['fare_amount'] < lower_bound) | (df_clean['fare_amount'] > upper_bound)]
print("Number of outliers in fare_amount:", len(outliers))

# Visualizations: Fare distribution
sns.histplot(df_clean['fare_amount'], bins=50)
plt.title('Fare Amount Distribution')
plt.savefig('fare_distribution.png')
plt.clf()

sns.boxplot(x=df_clean['fare_amount'])
plt.title('Fare Amount Outliers')
plt.savefig('fare_boxplot.png')
plt.clf()

# Convert pickup datetime to datetime format
df_clean['pickup_datetime'] = pd.to_datetime(df_clean['pickup_datetime'])

# Create new features: hour, day, month, weekday, period
df_clean['hour'] = df_clean['pickup_datetime'].dt.hour
df_clean['day'] = df_clean['pickup_datetime'].dt.day
df_clean['month'] = df_clean['pickup_datetime'].dt.month
df_clean['weekday'] = df_clean['pickup_datetime'].dt.dayofweek
df_clean['period'] = df_clean['hour'].apply(lambda x: 'Peak' if 7 <= x <= 9 or 17 <= x <= 20 else 'Off-Peak')

# Define Haversine function to calculate distance in km
def haversine(row):
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

# Optional: Filter out unrealistic distances (e.g., zero or > 100 km)
df_clean = df_clean[(df_clean['distance_km'] > 0) & (df_clean['distance_km'] < 100)]

# Visualize Fare vs Distance
sns.scatterplot(x='distance_km', y='fare_amount', data=df_clean)
plt.title("Fare Amount vs Distance (km)")
plt.xlabel("Distance (km)")
plt.ylabel("Fare Amount ($)")
plt.savefig('fare_vs_distance.png')
plt.clf()

# Visualize Fare vs Hour of Day
sns.boxplot(x='hour', y='fare_amount', data=df_clean)
plt.title('Fare Amount by Hour of Day')
plt.savefig('fare_by_hour.png')
plt.clf()

# Correlation heatmap
corr = df_clean[['fare_amount', 'hour', 'day', 'month', 'weekday', 'distance_km']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.clf()

# Export cleaned and enhanced dataset
df_clean.to_csv('uber_enhanced.csv', index=False)
