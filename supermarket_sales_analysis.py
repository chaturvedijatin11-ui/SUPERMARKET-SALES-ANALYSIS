# Supermarket Sales Data Science Project
# Author: Data Science Enthusiast
# Description: An end-to-end walkthrough analyzing 2,000 supermarket transactions, 
#              uncovering trends, and segmenting customers using K-Means clustering.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Setup our plotting vibe
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print("--- 1. LOADING UP THE DATA ---")
# Load our 2000-row supermarket dataset
df = pd.read_csv('supermarket_sales_2000.csv')
print(f"Awesome! Loaded {df.shape[0]} rows and {df.shape[1]} columns successfully.\n")

print("--- 2. CLEANING & FEATURE ENGINEERING ---")
# Convert types and pull out useful temporal features
df['Date'] = pd.to_datetime(df['Date'])
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
df['Month'] = df['Date'].dt.month_name()
df['Day_Name'] = df['Date'].dt.day_name()
print("Dates parsed and hours extracted. Ready for crunching!\n")

print("--- 3. EXPLORATORY DATA ANALYSIS (EDA) ---")
# Let's see which product lines bring in the most cash
product_sales = df.groupby('Product line')['Total'].sum().reset_index().sort_values(by='Total', ascending=False)
print("Top Product Lines by Total Revenue:")
print(product_sales.to_string(index=False))

# Plot 1: Product Line Revenue
plt.figure(figsize=(10, 5))
sns.barplot(data=product_sales, x='Total', y='Product line', palette='Blues_r')
plt.title('Total Sales by Product Line', fontsize=14, fontweight='bold')
plt.xlabel('Total Sales ($)', fontsize=12)
plt.ylabel('Product Line', fontsize=12)
plt.tight_layout()
plt.savefig('product_sales_report.png')
plt.close()

# Plot 2: Hourly Shopping Patterns
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Hour', palette='viridis')
plt.title('Transaction Count by Hour of the Day', fontsize=14, fontweight='bold')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
plt.tight_layout()
plt.savefig('hourly_sales_report.png')
plt.close()
print("\nSaved visual charts: 'product_sales_report.png' and 'hourly_sales_report.png'.\n")

print("--- 4. UNSUPERVISED LEARNING: CUSTOMER SEGMENTATION ---")
# Let's group transactions using K-Means based on unit price, quantity, and total spend
features = df[['Unit price', 'Quantity', 'Total']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Run K-Means with 3 natural clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Customer_Cluster'] = kmeans.fit_predict(scaled_features)

# Analyze the clusters
cluster_insights = df.groupby('Customer_Cluster')[['Unit price', 'Quantity', 'Total', 'Rating']].mean().reset_index()
print("Cluster Profiles Discovered:")
print(cluster_insights.to_string(index=False))

print("\n--- 5. WRAP UP ---")
print("Project pipeline completed smoothly! Check out the generated PNG charts and explore the cluster insights.")
