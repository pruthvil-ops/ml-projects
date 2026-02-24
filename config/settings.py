# Configuration settings for CIC-IDS2017 dataset
import os

# Dataset configuration
DATASET_CONFIG = {
    'target_column': 'Label',
    'sample_fraction': 0.1,  # Fraction of data to use for training
    'important_features_count': 30,
    'test_size': 0.2,
    'random_state': 42
}

# Model configuration
MODEL_CONFIG = {
    'lightgbm': {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    },
    'randomforest': {
        'n_estimators': 50,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Path configuration
PATHS = {
    'data_dir': 'data',
    'model_dir': 'model',
    'output_dir': 'output'
}

# Create directories
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)