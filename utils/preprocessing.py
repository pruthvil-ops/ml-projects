import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def split_features_target(df, target_column='Label', important_features=None):
    """
    Split dataframe into features and target for network anomaly detection
    """
    
    if target_column not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(f"Target column '{target_column}' not found. Available: {available_cols}")
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Remove identifier columns
    id_columns = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    for col in id_columns:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    # If important features are provided, use only those
    if important_features:
        available_important = [f for f in important_features if f in X.columns]
        X = X[available_important]
        print(f"Using {len(available_important)} important features")
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # For CIC-IDS2017, most columns should be numeric
    # Convert potential numeric columns stored as object
    for col in categorical_cols:
        if col in ['Protocol']:  # Known categorical
            continue
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if col in categorical_cols:
                categorical_cols.remove(col)
                numeric_cols.append(col)
        except:
            pass
    
    print(f"Target: {target_column}, Classes: {y.nunique()}")
    print(f"Features: {X.shape[1]} (Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)})")
    
    return X, y, numeric_cols, categorical_cols

def build_lightweight_preprocessor(numeric_cols, categorical_cols):
    """
    Build memory-efficient preprocessing pipeline
    """
    # Simple imputation for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # For categorical, use simple one-hot with limited categories
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        n_jobs=-1  # Use all available cores
    )
    
    return preprocessor

def preprocess_target(y):
    """Preprocess target labels efficiently"""
    # Encode labels if they're strings
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_mapping = {label: int(index) for index, label in enumerate(le.classes_)}
        print(f"Label mapping: {class_mapping}")
        return y_encoded, le
    return y, None