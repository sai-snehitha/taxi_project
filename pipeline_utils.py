import sys
import os
import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import hopsworks

# Ensure the `src` folder is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


# Function to get Hopsworks project
def get_hopsworks_project():
    """
    Returns the Hopsworks project instance.
    """
    project = hopsworks.login()
    return project


# Function to calculate the average rides over the last 4 weeks
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    last_4_weeks_columns = [
        f"rides_t-{7*24}",  # 1 week ago
        f"rides_t-{14*24}",  # 2 weeks ago
        f"rides_t-{21*24}",  # 3 weeks ago
        f"rides_t-{28*24}",  # 4 weeks ago
    ]

    # Ensure the required columns exist in the DataFrame
    missing_cols = [col for col in last_4_weeks_columns if col not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Calculate the average of the last 4 weeks
    X["average_rides_last_4_weeks"] = X[last_4_weeks_columns].mean(axis=1)

    return X


# FunctionTransformer to add the average rides feature
add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)


# Custom transformer to add temporal features
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_ = X.copy()
        if "pickup_hour" in X_.columns:
            X_["hour"] = X_["pickup_hour"].dt.hour
            X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek
            X_.drop(columns=["pickup_hour"], inplace=True)

        if "pickup_location_id" in X_.columns:
            X_.drop(columns=["pickup_location_id"], inplace=True)

        return X_


# Instantiate the temporal feature engineer
add_temporal_features = TemporalFeatureEngineer()


# Function to return the pipeline
def get_pipeline(**hyper_params):
    """
    Returns a pipeline with optional parameters for LGBMRegressor.

    Parameters:
    ----------
    **hyper_params : dict
        Optional parameters to pass to the LGBMRegressor.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline with feature engineering and LGBMRegressor.
    """
    # Move imports inside the function to prevent circular import issues
    from sklearn.pipeline import make_pipeline

    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyper_params),  # Pass optional parameters here
    )
    return pipeline
