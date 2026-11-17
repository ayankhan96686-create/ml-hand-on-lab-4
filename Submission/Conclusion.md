# Lab Evaluation 4: Conclusion

**Topic:** Module IV - Clustering & Association Rule Mining  
**Dataset:** Mall Customer Segmentation & Groceries Market Basket Analysis  

---

### Short Conclusion (150–200 Words)

In this lab evaluation, unsupervised clustering and association rule mining algorithms were successfully implemented on real-world retail datasets. For customer segmentation on the Mall Customers dataset, the Elbow Method identified $k = 5$ as the optimal number of clusters based on Annual Income and Spending Score. Both K-Means and Hierarchical Agglomerative clustering produced well-separated, actionable customer personas—Careless Spenders, Careful Savers, Sensible Budget Shoppers, Affluent VIPs, and Standard Customers—achieving strong Silhouette scores (~0.555). In comparison, DBSCAN effectively detected dense cluster cores while identifying 23 outlier points as noise (-1), showing higher robustness against irregular distributions.

For market basket analysis on the Groceries dataset, transactions grouped by customer ID and purchase date were mined using the Apriori and FP-Growth algorithms at 0.5% support and 10% confidence. Both algorithms discovered the exact same 126 frequent itemsets and 19 association rules. However, FP-Growth executed approximately 2.3 times faster than Apriori because its compressed FP-tree eliminates expensive iterative candidate generation. Strong cross-selling associations (such as frankfurters with vegetables and milk) provide actionable insights for retail shelf organization and promotional product bundling.
