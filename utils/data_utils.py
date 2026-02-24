import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def get_dataset_info(data_dir):
    """Get information about CSV files in the dataset"""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    total_size = 0
    file_info = []
    
    for f in csv_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        total_size += size_mb
        file_info.append({
            'file': os.path.basename(f),
            'size_mb': round(size_mb, 2)
        })
    
    return csv_files, file_info, total_size

def smart_sample_data(data_dir, target_column='Label', sample_frac=0.1, random_state=42):
    """
    Smart sampling from large dataset - preserves class distribution
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    sampled_dfs = []
    
    for file_path in csv_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        try:
            # Read the file with optimized parameters
            chunk = pd.read_csv(
                file_path, 
                encoding='latin1',
                low_memory=False,
                nrows=100000  # Limit rows per file for sampling
            )
            
            # Clean the chunk
            chunk = clean_dataframe(chunk, target_column)
            
            if chunk is not None and len(chunk) > 0:
                # Sample proportionally
                sample_size = max(1, int(len(chunk) * sample_frac))
                sampled_chunk = chunk.sample(n=min(sample_size, 50000), random_state=random_state)
                sampled_dfs.append(sampled_chunk)
                print(f"  Sampled {len(sampled_chunk)} rows")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    if sampled_dfs:
        combined_sample = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Total sampled data: {combined_sample.shape}")
        return combined_sample
    else:
        raise ValueError("No data could be sampled from the files")

def clean_dataframe(df, target_column):
    """Clean dataframe - remove invalid rows and columns"""
    if df is None or len(df) == 0:
        return None
        
    # Remove rows where target is missing
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])
    
    # Remove columns with too many missing values
    missing_threshold = 0.8
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

def get_feature_importance_ranking(data_dir, target_column='Label', top_k=30):
    """
    Quick feature importance analysis to select most important features
    """
    # Take a small sample for feature analysis
    sample_data = smart_sample_data(data_dir, target_column, sample_frac=0.05)
    
    from utils.preprocessing import split_features_target, preprocess_target
    
    X, y, numeric_cols, categorical_cols = split_features_target(sample_data, target_column)
    y_processed, _ = preprocess_target(y)
    
    # Use only numeric features for quick analysis
    X_numeric = X[numeric_cols].fillna(0)
    
    # Quick Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_numeric, y_processed)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': numeric_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(top_k)['feature'].tolist()
    print(f"Selected top {len(top_features)} features based on importance")
    
    return top_features