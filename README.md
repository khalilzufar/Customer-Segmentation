# Customer Segmentation Analysis

A machine-learning project that groups credit-card customers by purchasing behavior and account characteristics.

## Objective

Identify customer segments that can support more targeted marketing, retention, and customer-experience decisions.

## Dataset

The notebook uses credit_card.csv, containing 4,475 customer records and 18 behavioral or account-level features.

## Workflow

1. Explore missing values, distributions, skewness, and relationships between features.
2. Impute missing values and cap outliers.
3. Standardize numerical features.
4. Reduce dimensionality with PCA.
5. Compare candidate cluster counts using the Elbow Method and Silhouette Score.
6. Train K-Means with three clusters.
7. Profile the resulting clusters and save the preprocessing/model artifacts.

## Model and evaluation

The final notebook uses six PCA components and KMeans with three clusters. The recorded average silhouette scores are:

| Clusters | Silhouette score |
| ---: | ---: |
| 2 | 0.313 |
| 3 | 0.288 |
| 4 | 0.261 |

Three clusters were retained for the project analysis; the scores also show why cluster selection should be validated with business context, not one metric alone.

## Repository contents

- customer_segmentation.ipynb — complete EDA, preprocessing, clustering, profiling, and inference workflow.
- credit_card.csv — input dataset.
- scaler.pkl, pca.pkl, kmeans.pkl — saved preprocessing and clustering artifacts.
- list_num_col.txt — numerical feature list used by inference.

## Run locally

~~~bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn feature-engine jupyter
jupyter notebook customer_segmentation.ipynb
~~~

## Author

[Khalil Zufar](https://github.com/khalilzufar)
