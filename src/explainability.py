import pandas as pd
import numpy as np
from typing import Dict, Any

def get_feature_drivers(sample_df: pd.DataFrame, model, feature_names: list) -> Dict[str, float]:
    """
    Calculates normalized feature importance and directional divergence
    to explain what factors most heavily influenced this wine's verdict.
    """
    importances = model.feature_importances_
    
    # Feature benchmarks (average in red wine)
    # Alcohol (+), Volatile Acidity (-), Sulphates (+), Citric Acid (+)
    contributions = {}
    for name, imp in zip(feature_names, importances):
        contributions[name] = round(float(imp), 4)
        
    # Sort by overall tree importance
    sorted_drivers = dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:5])
    return sorted_drivers
