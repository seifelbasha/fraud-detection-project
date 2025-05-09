Credit Card Fraud Detection and Analysis
Graduation Project – DEPI Initiative
Ministry of Communications and Information Technology

🔍 Overview
This machine learning project focuses on detecting fraudulent credit card transactions using structured transactional data. The end-to-end pipeline covers data preprocessing, feature engineering, unsupervised clustering, model training, evaluation, and visualization.

💾 Dataset
The dataset contains anonymized credit card transactions along with metadata such as transaction amount, time, customer behavior flags, and fraud labels. The goal is to identify fraudulent patterns and build a reliable fraud detection system.

📌 Objective
The primary objective is to predict whether a transaction is fraudulent, based on historical data and engineered features, while analyzing customer and terminal behavior patterns.

🧩 Project Milestones
Data Preprocessing & Cleaning

Converted timestamp columns to datetime format

Created derived features (e.g. transaction hour, weekday, is_weekend)

Handled missing and duplicate values

Feature Engineering

Engineered behavioral flags (e.g. IS_NIGHT, IS_WEEKEND, NEW_TERMINAL_FLAG)

Calculated rolling averages, z-scores, and standard deviations

Created customer-level and terminal-level aggregations

Clustering & Unsupervised Analysis

Applied PCA for dimensionality reduction

Tested clustering methods: KMeans, DBSCAN, Spectral Clustering, Agglomerative Clustering

Visualized cluster distributions and cluster-based fraud ratios

Model Development & Evaluation

Built supervised models for classification (e.g. Logistic Regression, Random Forest, XGBoost)

Evaluated models using confusion matrix, precision, recall, F1-score, and ROC-AUC

Applied threshold tuning for fraud probability cutoffs

Visualization & Reporting

Used seaborn and matplotlib to visualize transaction distributions and fraud patterns

Count plots, heatmaps, and temporal patterns

Correlation analysis of key features

⚙️ Getting Started
To run the project locally:
