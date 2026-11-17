# Machine Learning Lab Evaluation 4 - Clustering & Association Rule Mining

This repository contains the complete implementation for **Hands-on Lab Evaluation 4 (Module IV)** covering unsupervised machine learning and market basket analysis.

---

## 📌 Objectives & Tasks

1. **Task 1 - Data Exploration:**
   - Load `Mall_Customers.csv` using Pandas.
   - Inspect first/last 10 records, dataset info, summary statistics, and missing values.
   - Visualize feature distributions (Age, Annual Income, Spending Score, Gender).

2. **Task 2 - Data Preprocessing:**
   - Handle missing values (if any).
   - Select key behavioral features (`Annual Income (k$)` and `Spending Score (1-100)`).
   - Scale numerical features using `StandardScaler`.

3. **Task 3 - K-Means Clustering:**
   - Find optimal number of clusters using the **Elbow Method** (WCSS / Inertia).
   - Train K-Means with optimal $k = 5$.
   - Calculate Silhouette Score and plot clusters with centroids.

4. **Task 4 - Hierarchical Clustering & DBSCAN:**
   - Plot Dendrogram using Ward's linkage.
   - Train Agglomerative Hierarchical Clustering with $k = 5$ and compare with K-Means.
   - Implement DBSCAN (density-based clustering), identify core clusters and outlier/noise points.
   - Generate side-by-side visual comparison of all 3 algorithms.

5. **Task 5 - Groceries Market Basket Analysis (Apriori vs FP-Growth):**
   - Load `Groceries_dataset.csv` and group purchases by member and date into shopping transactions.
   - Run **Apriori** and **FP-Growth** algorithms at `min_support = 0.005` (0.5%) and `min_confidence = 0.10` (10%).
   - Compare execution time, frequent itemsets found, and rules generated.
   - Save top association rules sorted by Lift.

---

## 📂 Project Structure

```text
├── Mall_Customers.csv               # Mall customer dataset
├── Groceries_dataset.csv            # Retail transaction dataset
├── lab4_code.py                     # Clean Python script (runs all 5 tasks)
├── lab4_clustering_association.ipynb # Jupyter Notebook with visual outputs
├── association_rules_report.csv     # Generated association rules report
├── output_graphs/                   # Saved visualization figures
│   ├── 1_feature_distributions.png
│   ├── 2_kmeans_elbow_method.png
│   ├── 3_kmeans_clusters.png
│   ├── 4_hierarchical_dendrogram.png
│   ├── 5_hierarchical_clusters.png
│   ├── 6_dbscan_clusters.png
│   └── 7_clustering_comparison.png
└── README.md                        # Documentation & setup guide
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
```

### 2. Run Python Script
```bash
python lab4_code.py
```

All 7 visualization graphs will be saved into the `output_graphs/` folder, and the mined association rules will be exported to `association_rules_report.csv`.

### 3. Or Open the Jupyter Notebook
Open `lab4_clustering_association.ipynb` in Jupyter Notebook, VS Code, or Google Colab and run all cells sequentially.

---

## 📊 Summary of Results

### Clustering Performance Comparison
| Algorithm | No. of Clusters | Silhouette Score | Outlier Handling | Shape / Behavior |
| :--- | :---: | :---: | :--- | :--- |
| **K-Means** | 5 | **0.5547** | Forces all points into clusters | Spherical clusters |
| **Hierarchical (Ward)** | 5 | **0.5538** | Forces all points into clusters | Tree hierarchy |
| **DBSCAN** | 6 | **0.5577** | Isolated **23 points as noise (-1)** | Density-based |

### Association Rule Mining Comparison
| Metric | Apriori | FP-Growth | Observation |
| :--- | :---: | :---: | :--- |
| **Execution Time** | ~0.11 s | ~0.05 s | **FP-Growth is ~2.3x faster** |
| **Frequent Itemsets** | 126 | 126 | Exact same itemsets discovered |
| **Rules Mined** | 19 | 19 | Exact same rules generated |

---

## 📝 Conclusion

In this lab, unsupervised clustering and association rule mining were implemented on real-world retail datasets. For customer segmentation on the Mall Customers dataset, the Elbow Method identified $k = 5$ as the optimal number of clusters based on Annual Income and Spending Score. K-Means and Hierarchical Agglomerative clustering produced well-separated, actionable customer personas (Spendthrifts, Careful Savers, Budget Shoppers, Affluent VIPs, and Standard Customers) with high Silhouette scores (~0.555). In contrast, DBSCAN identified dense clusters while isolating 23 outlier points as noise, demonstrating superior robustness against anomalies.

For market basket analysis on the Groceries dataset, transactions grouped by member and date were analyzed using Apriori and FP-Growth algorithms at 0.5% support and 10% confidence. Both algorithms identified 126 frequent itemsets and 19 association rules. However, FP-Growth executed roughly 2.3 times faster than Apriori because its compressed FP-tree eliminates expensive iterative candidate generation. Strong cross-purchasing associations were discovered between meats, bakery goods, and dairy products. These insights allow retailers to optimize targeted marketing, shelf placements, and promotional product bundles effectively.
