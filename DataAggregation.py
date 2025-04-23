import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def ensure_dataframe(data):
    """Convert numpy array to DataFrame if needed."""
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape(-1, 1)  # Convert 1D arrays to 2D
        return pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise ValueError("Input must be a pandas DataFrame or a NumPy array")

def convert_numpy(obj):
    """Convert NumPy data types to native Python types."""
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    return obj

def has_numeric_value(data):
    df = ensure_dataframe(data)
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        return False
    else:
        return True

def get_data_size(data):
    df = ensure_dataframe(data)
    return df.size
    
def summary_statistics(data):
    df = ensure_dataframe(data)
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        return None
    return numeric_df.describe().round(2).to_dict()

def data_distribution(data):
    df = ensure_dataframe(data)
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        return None
    return {
        'skewness': numeric_df.apply(lambda x: round(skew(x), 2) if np.std(x) > 1e-6 else np.nan).to_dict(),
        'kurtosis': numeric_df.apply(lambda x: round(kurtosis(x), 2) if np.std(x) > 1e-6 else np.nan).to_dict()
    }

def feature_correlation(data):
    df = ensure_dataframe(data)
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        return None
    return numeric_df.corr(method='pearson').round(2).to_dict()

def categorical_feature_analysis(data):
    """Analyze categorical features by counting unique values and most frequent category."""
    df = ensure_dataframe(data)
    categorical_df = df.select_dtypes(include=['object'])
    if categorical_df.empty:
        return None
    return {
        col: {
            "unique_values": categorical_df[col].nunique(),
            "most_frequent_value": categorical_df[col].mode()[0] if not categorical_df[col].mode().empty else None,
            "missing_values": categorical_df[col].isnull().sum()
        } 
        for col in categorical_df.columns
    }

def missing_value_summary(data):
    """Summarize missing values for all features."""
    df = ensure_dataframe(data)
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df) * 100).round(2)

    # Properly filter columns with missing values
    missing_info = {
        col: {"missing_count": int(count), "missing_percentage": float(missing_percentages[col])}
        for col, count in missing_counts.items() if count > 0
    }

    return missing_info if missing_info else {"message": "No missing values"}


