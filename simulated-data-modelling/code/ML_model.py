# %% [markdown]
# # Fraud Detection Pipeline with Time-Aware Validation
# 
# This script implements a fraud detection pipeline with strict temporal controls and anti-leakage measures.

# %%
# Standard imports
import os
import pandas as pd
import numpy as np
import math
import sys
import time
import pickle
import json
import datetime
import random
from collections import Counter

# Data handling and machine learning
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Model and resampling techniques
import xgboost
from imblearn.over_sampling import ADASYN

# Self-organizing maps
from minisom import MiniSom

# Network analysis
import networkx as nx
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB  # Closest available to Bayesian approach
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Self-organizing maps
from minisom import MiniSom

# Network analysis
import networkx as nx

"""# Machine Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
"""
# Warnings
import warnings
warnings.filterwarnings('ignore')

# %%
# Load data and sort by datetime
df = pd.read_csv('simulated-data-raw1.csv')
df = df.sort_values('TX_DATETIME').reset_index(drop=True)

# %%
# SAFEGUARD: Data Validation
required_cols = ['CUSTOMER_ID', 'TX_DATETIME', 'TX_AMOUNT', 'TX_FRAUD']
assert set(required_cols).issubset(df.columns), "Missing critical columns"

# SAFEGUARD: Temporal Cohesion Check
assert df['TX_DATETIME'].is_monotonic_increasing, "Data not chronological"

# %%
def split_train_delay_test(df, start_date, train_days=7, delay_days=7, test_days=7):
    """
    Time-based split with rigorous anti-leakage controls
    
    Args:
        df: Input dataframe with transaction data
        start_date: Start date for the training period
        train_days: Duration of training period in days
        delay_days: Duration of delay period in days
        test_days: Duration of test period in days
        
    Returns:
        train_df, delay_df, test_df: Three splits with temporal isolation
    """
    # Ensure required columns exist
    required_cols = ['CUSTOMER_ID', 'TX_DATETIME', 'TX_FRAUD', 'TX_AMOUNT']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    # Convert start_date to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Ensure datetime dtype for transaction data
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df = df.sort_values('TX_DATETIME').reset_index(drop=True)
    
    # Define temporal boundaries
    train_start = start_date
    train_end = train_start + pd.DateOffset(days=train_days)
    delay_end = train_end + pd.DateOffset(days=delay_days)
    test_end = delay_end + pd.DateOffset(days=test_days)
    
    # Validate temporal sequence
    if df['TX_DATETIME'].max() < test_end:
        raise ValueError("Test period extends beyond available data")
    
    # --- Core Split Logic ---
    # Training Set: Initial period
    train_mask = df['TX_DATETIME'].between(train_start, train_end, inclusive='left')
    train_df = df.loc[train_mask].copy()
    
    # Delay Set: Gap period for fraud pattern observation
    delay_mask = df['TX_DATETIME'].between(train_end, delay_end, inclusive='left')
    delay_df = df.loc[delay_mask].copy()
    
    # Test Set: Future period with leakage protection
    test_mask = df['TX_DATETIME'].between(delay_end, test_end, inclusive='left')
    test_df = df.loc[test_mask].copy()
    
    # --- Fraud Exclusion Protocol ---
    # Identify fraudsters observed in training OR delay periods
    train_frauds = set(train_df.loc[train_df['TX_FRAUD'] == 1, 'CUSTOMER_ID'])
    delay_frauds = set(delay_df.loc[delay_df['TX_FRAUD'] == 1, 'CUSTOMER_ID'])
    known_frauds = train_frauds.union(delay_frauds)
    
    # Remove transactions from known fraudsters in test set
    test_df = test_df[~test_df['CUSTOMER_ID'].isin(known_frauds)]
    
    # --- Temporal Validation ---
    # Ensure no chronological overlap
    assert train_df['TX_DATETIME'].max() <= delay_df['TX_DATETIME'].min(), "Training/Delay overlap"
    assert delay_df['TX_DATETIME'].max() <= test_df['TX_DATETIME'].min(), "Delay/Test overlap"
    
    return train_df, delay_df, test_df

# %%
# Apply the time-based split
train, delay, test = split_train_delay_test(df, start_date='2024-04-01')

# %%
def calculate_customer_features(group):
    """Calculate customer-specific behavioral features"""
    # Sort transactions chronologically
    group = group.sort_values('TX_DATETIME')
    
    # Expanding window statistics
    group['CUSTOMER_TX_FREQ'] = range(1, len(group)+1)
    group['CUSTOMER_AVG_AMOUNT'] = group['TX_AMOUNT'].expanding().mean()
    group['CUSTOMER_STD'] = group['TX_AMOUNT'].expanding().std()
    
    # Handle NaN values in standard deviation
    group['Z_SCORE'] = (group['TX_AMOUNT'] - group['CUSTOMER_AVG_AMOUNT']) / group['CUSTOMER_STD']
    
    # Time since last transaction
    group['TIME_SINCE_LAST_TX'] = group['TX_DATETIME'].diff().dt.total_seconds().fillna(-1)
    
    group['Z_SCORE'] = group['Z_SCORE'].fillna(0)  # Or use -1 if meaningful
    group['TIME_SINCE_LAST_TX'] = group['TIME_SINCE_LAST_TX'].fillna(-1)
    
    return group

# Apply to each dataset in order
train = train.groupby('CUSTOMER_ID', group_keys=False).apply(calculate_customer_features)
delay = pd.concat([train, delay]).groupby('CUSTOMER_ID', group_keys=False).apply(calculate_customer_features)
test = pd.concat([train, delay]).groupby('CUSTOMER_ID', group_keys=False).apply(calculate_customer_features)

# %%
def time_window_features(group, lookback='1H'):
    """Calculate time-windowed transaction features"""
    # Validate required columns
    mandatory_cols = ['CUSTOMER_ID', 'TX_DATETIME', 'TX_AMOUNT']
    missing = [col for col in mandatory_cols if col not in group.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Ensure datetime type
    group = group.copy()
    group['TX_DATETIME'] = pd.to_datetime(group['TX_DATETIME'])
    
    # Sort before windowing
    group = group.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
    
    # Calculate rolling features with unique column names
    rolling_df = (
        group.set_index('TX_DATETIME')
        .groupby('CUSTOMER_ID')['TX_AMOUNT']
        .rolling(lookback)
        .agg(['count', 'sum'])
        .reset_index()
    )
    
    # Rename columns with unique suffixes
    rolling_df = rolling_df.rename(columns={
        'count': f'TXs_LAST_{lookback}',
        'sum': f'AMOUNT_LAST_{lookback}'
    })
    
    # Drop existing rolling feature columns if they exist
    for col in [f'TXs_LAST_{lookback}', f'AMOUNT_LAST_{lookback}']:
        if col in group.columns:
            group = group.drop(columns=col)
    
    rolling_df = rolling_df.fillna(0)  # Replace NaNs with 0 for no history
    
    # Merge back with original data
    result = pd.merge(
        group,
        rolling_df,
        on=['CUSTOMER_ID', 'TX_DATETIME'],
        how='left',
        suffixes=('', '_DROP')  # Explicit suffix for potential duplicates
    )
    
    # Drop any columns that ended up with _DROP suffix
    result = result[[col for col in result.columns if not col.endswith('_DROP')]]
    
    return result

# Apply to each dataset with proper chaining
train = time_window_features(train)
delay = time_window_features(pd.concat([train, delay]))
test = time_window_features(pd.concat([train, delay, test]))

# %%
def terminal_features(group, historical_data=None):
    """Calculate terminal-related features"""
    if historical_data is None:
        historical_data = group
    
    # Verify required columns exist
    required_cols = ['TERMINAL_ID', 'CUSTOMER_ID', 'TX_DATETIME']
    missing = [col for col in required_cols if col not in group.columns]
    if missing:
        raise ValueError(f"Missing columns for terminal features: {missing}")
    
    # Terminal usage patterns
    terminal_stats = historical_data.groupby('TERMINAL_ID').agg(
        TERMINAL_FREQ=('TX_DATETIME', 'count'),
        TERMINAL_DEGREE=('CUSTOMER_ID', 'nunique')
    ).reset_index()
    
    # Merge features
    result = group.merge(
        terminal_stats,
        on='TERMINAL_ID', 
        how='left'
    )
    
    # Customer-terminal interaction
    customer_terminal_stats = historical_data.groupby(
        ['CUSTOMER_ID', 'TERMINAL_ID']
    ).agg(
        CUSTOMER_TERMINAL_FREQ=('TX_DATETIME', 'count')
    ).reset_index()
    
    result = result.merge(
        customer_terminal_stats,
        on=['CUSTOMER_ID', 'TERMINAL_ID'],
        how='left'
    )
    
    # New terminal flag
    customer_terminal_map = historical_data.groupby('CUSTOMER_ID')['TERMINAL_ID'].apply(set).to_dict()
    result['NEW_TERMINAL_FLAG'] = result.apply(
        lambda row: int(row['TERMINAL_ID'] not in customer_terminal_map.get(row['CUSTOMER_ID'], set())),
        axis=1
    )
    
    return result

# Apply to each dataset
train = terminal_features(train)
delay = terminal_features(delay, pd.concat([train, delay]))
test = terminal_features(test, pd.concat([train, delay]))

# %%
# Convert timestamp column to datetime type
train['TX_DATETIME'] = pd.to_datetime(train['TX_DATETIME'])
delay['TX_DATETIME'] = pd.to_datetime(delay['TX_DATETIME']) 
test['TX_DATETIME'] = pd.to_datetime(test['TX_DATETIME'])

# %%
# Extract temporal features
df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
df['TX_DAY'] = df['TX_DATETIME'].dt.dayofweek
df['TX_IS_WEEKEND'] = df['TX_DAY'].isin([5, 6]).astype(int)
df['TX_HOUR_CATEGORY'] = pd.cut(
    df['TX_HOUR'], 
    bins=[0, 6, 12, 18, 24], 
    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
    right=False
)

# %%
# Resample only the training data:
from imblearn.over_sampling import ADASYN

# Prepare features and targets
train['TX_HOUR'] = train['TX_DATETIME'].dt.hour
test['TX_HOUR'] = test['TX_DATETIME'].dt.hour
train['TX_DAY'] = train['TX_DATETIME'].dt.dayofweek
test['TX_DAY'] = test['TX_DATETIME'].dt.dayofweek

# Feature selection - REMOVED ID COLUMNS
feature_cols = ['TX_AMOUNT', 'TERMINAL_DEGREE', 'TXs_LAST_1H', 'Z_SCORE', 'TIME_SINCE_LAST_TX', 'NEW_TERMINAL_FLAG']

"""feature_cols = [
    'NEW_TERMINAL_FLAG', 'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'TX_DAY', 'CUSTOMER_STD', 'TX_HOUR',
    'AMOUNT_TO_AVG_MONTH', 'AMOUNT_TO_AVG_WEEK', 'AMOUNT_TO_AVG_DAY', 'AMOUNT_TO_AVG',
    'CUSTOMER_AVG_AMOUNT', 'AMOUNT_LAST_1H', 'TIME_SINCE_LAST_TERMINAL_USE',
    'CUSTOMER_TERMINAL_FREQ', 'CUSTOMER_TX_FREQ', 'TX_IS_WEEKEND', 'TX_HOUR_CATEGORY'
]"""

X_train = train[feature_cols]
X_test = test[feature_cols]
y_train = train['TX_FRAUD']
y_test = test['TX_FRAUD']

# Apply ADASYN to training data only
ada = ADASYN(random_state=42)
X_train_res, y_train_res = ada.fit_resample(X_train, y_train)
X_train_res = pd.DataFrame(X_train_res, columns=feature_cols)

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Identify numerical/categorical columns
num_cols = ['TX_AMOUNT', 'Z_SCORE', 'TIME_SINCE_LAST_TX', 'TXs_LAST_1H', 'AMOUNT_LAST_1H', 'NEW_TERMINAL_FLAG']

# Preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define models and parameter grids
model_configs = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]),
        "params": {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        }
    },
    
    "Random Forest": {
        "pipeline": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ]),
        "params": {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5, 10]
        }
    },
    
    "Naive Bayes": {
        "pipeline": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GaussianNB())
        ]),
        "params": {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    },
    
    "Decision Tree": {
        "pipeline": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42))
        ]),
        "params": {
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__criterion': ['gini', 'entropy']
        }
    }
    ,
    
    "XGBoost": {
    "pipeline": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        ))
    ]),
    "params": {
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5],
        'classifier__n_estimators': [100, 200]
    }
}
}

# Store results
results = {}

# Define temporal cross-validator
tscv = TimeSeriesSplit(n_splits=5)

# Train and evaluate each model
for model_name, config in model_configs.items():
    print(f"\n=== Tuning {model_name} ===")
    
    # Initialize GridSearch
    grid_search = GridSearchCV(
        config["pipeline"],
        config["params"],
        cv=tscv,
        scoring='average_precision',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit model
    start_time = time.time()
    grid_search.fit(X_train_res, y_train_res)
    duration = time.time() - start_time
    
    # Get predictions
    y_pred = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
    
    # Store results
    results[model_name] = {
        "best_params": grid_search.best_params_,
        "cv_score": grid_search.best_score_,
        "test_ap": average_precision_score(y_test, y_pred),
        "test_auc": roc_auc_score(y_test, y_pred),
        "duration": duration
    }
    
    # Print intermediate results
    print(f"Best params: {grid_search.best_params_}")
    print(f"CV AP: {results[model_name]['cv_score']:.3f}")
    print(f"Test AP: {results[model_name]['test_ap']:.3f}")
    print(f"Test AUC: {results[model_name]['test_auc']:.3f}")
    print(f"Training time: {results[model_name]['duration']:.1f}s")

# %%
# Final comparison
print("\n=== Final Comparison ===")
comparison_df = pd.DataFrame(results).T
comparison_df[['cv_score', 'test_ap', 'test_auc', 'duration']].sort_values('test_ap', ascending=False)
print(comparison_df)