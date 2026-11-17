# Hands-on Lab Evaluation - 4
# Module IV: Clustering & Association Rule Mining
# Student Code: Mall Customers & Groceries Market Basket Analysis

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Change directory to script folder so outputs always save inside Submission/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create folder to save output plots
os.makedirs("output_graphs", exist_ok=True)

# --------------------------------------------------
# Task 1: Load and explore the Mall Customer dataset from GitHub
# --------------------------------------------------
# Load dataset directly from raw GitHub link
mall_url = "https://raw.githubusercontent.com/ayankhan96686-create/ml-hand-on-lab-4/main/Mall_Customers.csv"
try:
    mall = pd.read_csv(mall_url)
    print("Loaded Mall Customers dataset from GitHub successfully!")
except Exception:
    mall = pd.read_csv("Mall_Customers.csv")
    print("Loaded Mall Customers dataset from local file.")

print("\nFirst 10 rows:")
print(mall.head(10))

print("\nLast 10 rows:")
print(mall.tail(10))

print("\nDataset Info:")
mall.info()

print("\nSummary Statistics:")
print(mall.describe())

print("\nMissing values:")
print(mall.isnull().sum())

# Visualizing feature distributions
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
sns.histplot(mall['Age'], kde=True, color='skyblue')
plt.title("Age Distribution")

plt.subplot(2, 2, 2)
sns.histplot(mall['Annual Income (k$)'], kde=True, color='salmon')
plt.title("Annual Income (k$)")

plt.subplot(2, 2, 3)
sns.histplot(mall['Spending Score (1-100)'], kde=True, color='lightgreen')
plt.title("Spending Score (1-100)")

plt.subplot(2, 2, 4)
sns.countplot(data=mall, x='Gender', palette='Set2', hue='Gender', legend=False)
plt.title("Gender Count")

plt.tight_layout()
plt.savefig("output_graphs/1_feature_distributions.png")
plt.close()
print("Saved output_graphs/1_feature_distributions.png")

# --------------------------------------------------
# Task 2: Data Preprocessing and Feature Scaling
# --------------------------------------------------
# Check for null values
if mall.isnull().sum().sum() > 0:
    mall.dropna(inplace=True)

# We use Annual Income and Spending Score for customer segmentation
X = mall[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeatures scaled successfully. Shape:", X_scaled.shape)

# --------------------------------------------------
# Task 3: K-Means Clustering and Elbow Method
# --------------------------------------------------
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

# Plot the elbow curve
plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), wcss, marker='o', color='blue')
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS / Inertia")
plt.axvline(x=5, color='red', linestyle='--', label='Elbow at k=5')
plt.legend()
plt.tight_layout()
plt.savefig("output_graphs/2_kmeans_elbow_method.png")
plt.close()
print("Saved output_graphs/2_kmeans_elbow_method.png")

# Train KMeans with k=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
mall['KMeans_Cluster'] = kmeans_labels

km_score = silhouette_score(X_scaled, kmeans_labels)
print(f"K-Means (k=5) Silhouette Score: {km_score:.4f}")

# Plot KMeans clusters with labels
plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'green', 'purple', 'orange']
labels = [
    'Careless (Low Income, High Spend)',
    'Careful (High Income, Low Spend)',
    'Sensible (Low Income, Low Spend)',
    'Affluent (High Income, High Spend)',
    'Standard (Average Income & Spend)'
]

for i in range(5):
    plt.scatter(X[kmeans_labels == i, 0], X[kmeans_labels == i, 1], s=50, c=colors[i], label=labels[i])

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.title("Customer Segments using K-Means (k=5)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("output_graphs/3_kmeans_clusters.png")
plt.close()
print("Saved output_graphs/3_kmeans_clusters.png")

# --------------------------------------------------
# Task 4: Hierarchical Clustering and DBSCAN
# --------------------------------------------------
# Plot dendrogram
linkage_matrix = linkage(X_scaled, method='ward')
plt.figure(figsize=(8, 4))
dendrogram(linkage_matrix, truncate_mode='lastp', p=25)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Clusters")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.savefig("output_graphs/4_hierarchical_dendrogram.png")
plt.close()
print("Saved output_graphs/4_hierarchical_dendrogram.png")

# Agglomerative clustering
hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
hc_labels = hierarchical.fit_predict(X_scaled)
hc_score = silhouette_score(X_scaled, hc_labels)
print(f"Hierarchical Clustering Silhouette Score: {hc_score:.4f}")

# Plot hierarchical clusters
plt.figure(figsize=(7, 5))
for i in range(5):
    plt.scatter(X[hc_labels == i, 0], X[hc_labels == i, 1], s=50, c=colors[i], label=f"Cluster {i}")
plt.title("Hierarchical Clustering (k=5)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()
plt.savefig("output_graphs/5_hierarchical_clusters.png")
plt.close()
print("Saved output_graphs/5_hierarchical_clusters.png")

# DBSCAN
dbscan = DBSCAN(eps=0.35, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise_db = (db_labels == -1).sum()
print(f"DBSCAN clusters found: {n_clusters_db}, Outlier noise points: {n_noise_db}")

# Plot DBSCAN clusters
plt.figure(figsize=(7, 5))
for k in set(db_labels):
    if k == -1:
        plt.scatter(X[db_labels == k, 0], X[db_labels == k, 1], c='black', marker='x', s=60, label='Outliers / Noise')
    else:
        plt.scatter(X[db_labels == k, 0], X[db_labels == k, 1], s=50, label=f'Cluster {k}')
plt.title("DBSCAN Clustering (eps=0.35, min_samples=5)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("output_graphs/6_dbscan_clusters.png")
plt.close()
print("Saved output_graphs/6_dbscan_clusters.png")

# Side-by-side comparison plot
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
for i in range(5):
    plt.scatter(X[kmeans_labels == i, 0], X[kmeans_labels == i, 1], s=30)
plt.title(f"K-Means (k=5)\nScore: {km_score:.3f}")
plt.xlabel("Income")
plt.ylabel("Score")

plt.subplot(1, 3, 2)
for i in range(5):
    plt.scatter(X[hc_labels == i, 0], X[hc_labels == i, 1], s=30)
plt.title(f"Hierarchical (k=5)\nScore: {hc_score:.3f}")
plt.xlabel("Income")

plt.subplot(1, 3, 3)
for k in set(db_labels):
    if k == -1:
        plt.scatter(X[db_labels == k, 0], X[db_labels == k, 1], c='black', marker='x', s=30)
    else:
        plt.scatter(X[db_labels == k, 0], X[db_labels == k, 1], s=30)
plt.title(f"DBSCAN\nClusters: {n_clusters_db}, Noise: {n_noise_db}")
plt.xlabel("Income")

plt.tight_layout()
plt.savefig("output_graphs/7_clustering_comparison.png")
plt.close()
print("Saved output_graphs/7_clustering_comparison.png")

# --------------------------------------------------
# Task 5: Groceries Market Basket Analysis from GitHub
# --------------------------------------------------
groceries_url = "https://raw.githubusercontent.com/ayankhan96686-create/ml-hand-on-lab-4/main/Groceries_dataset.csv"
try:
    groceries = pd.read_csv(groceries_url)
    print("\nLoaded Groceries dataset from GitHub successfully!")
except Exception:
    groceries = pd.read_csv("Groceries_dataset.csv")
    print("\nLoaded Groceries dataset from local file.")

print("Groceries dataset sample:")
print(groceries.head())

# Group items by member and date into single transactions
baskets = groceries.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).tolist()
print("Total shopping baskets:", len(baskets))

# Encode baskets into binary matrix
te = TransactionEncoder()
te_data = te.fit(baskets).transform(baskets)
df_trans = pd.DataFrame(te_data, columns=te.columns_)

# 1. Apriori algorithm
t0 = time.time()
freq_items_apriori = apriori(df_trans, min_support=0.005, use_colnames=True)
apriori_time = time.time() - t0
rules_apriori = association_rules(freq_items_apriori, metric="confidence", min_threshold=0.10)

# 2. FP-Growth algorithm
t0 = time.time()
freq_items_fpgrowth = fpgrowth(df_trans, min_support=0.005, use_colnames=True)
fpgrowth_time = time.time() - t0
rules_fpgrowth = association_rules(freq_items_fpgrowth, metric="confidence", min_threshold=0.10)

print("\n--- Apriori vs FP-Growth Comparison ---")
print(f"Apriori Time:   {apriori_time:.4f} seconds | Itemsets: {len(freq_items_apriori)} | Rules: {len(rules_apriori)}")
print(f"FP-Growth Time: {fpgrowth_time:.4f} seconds | Itemsets: {len(freq_items_fpgrowth)} | Rules: {len(rules_fpgrowth)}")
print(f"Observation: FP-Growth is {apriori_time / fpgrowth_time:.2f}x faster than Apriori!")

# Print top rules sorted by lift
print("\nTop 10 Rules by Lift:")
top_rules = rules_apriori.sort_values(by="lift", ascending=False).head(10).copy()
top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))

# Save rules to CSV
top_rules.to_csv("association_rules_report.csv", index=False)
print("\nSaved rules to association_rules_report.csv")
print("\nDone! All tasks completed.")
