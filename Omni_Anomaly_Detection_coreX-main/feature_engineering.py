import pandas as pd
import numpy as np

def remove_constant_features(df, numeric_cols):
    """
    Removes columns that have zero variance (constant values across all rows),
    as they provide no useful information for the anomaly detection model.
    """
    print("--- [Feature Selection] Checking for constant features ---")
    
    # Check for constant columns only within the numeric columns
    constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
    
    if constant_cols:
        print(f"Dropped {len(constant_cols)} constant features: {constant_cols}")
        df = df.drop(columns=constant_cols)
    else:
        print("No constant features found.")
        
    return df

def remove_correlated_features(df, threshold=0.95):
    """
    Removes features that are highly correlated with others,
    retaining only one to reduce redundancy.
    """
    print(f"--- [Feature Selection] Checking for highly correlated features (>{threshold}) ---")
    
    # Compute correlation matrix for numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Select the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify columns to drop (those with correlation greater than the threshold)
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print("No highly correlated features found.")
        
    # Get remaining numeric columns to pass them forward in the pipeline
    remaining_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return df, remaining_numeric
