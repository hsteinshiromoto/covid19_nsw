from math import ceil, floor
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typeguard import typechecked


@typechecked
def get_cases_data() -> pd.DataFrame:
    """Get cases data from csv file

    Returns:
        (pd.DataFrame): Cases data
    """
    return pd.read_csv('https://data.nsw.gov.au/data/dataset/97ea2424-abaf-4f3e-a9f2-b5c883f42b6a/resource/2776dbb8-f807-4fb2-b1ed-184a6fc2c8aa/download/confirmed_cases_table4_location_likely_source.csv')


@typechecked
def to_datetime(column: str, data: pd.DataFrame)  -> pd.DataFrame:
    """Convert date column to datetime

    Returns:
        (pd.DataFrame): Cases data
    """

    data[column] = pd.to_datetime(data[column])

    return data


@typechecked
def subset_latest_outbreak(date: str, likely_source_of_infection: str
                        ,data: pd.DataFrame) -> pd.DataFrame:
    """Subset data to latest outbreak

    Args:
        date (str): Date to subset to
        likely_source_of_infection (str): Source of infection
        data (pd.DataFrame): Data to subset

    Returns:
        (pd.DataFrame): Subset data
    """
    return data.query(f"notification_date >= '{date}' & likely_source_of_infection != '{likely_source_of_infection}'")


@typechecked
def get_daily_cases_stats(data: pd.DataFrame
                        ,summarize_feature_name: str="Daily Number of Cases"
                        ,datetime_col_name: str="notification_date"
                        ,initial_number_of_cases: int=10) -> pd.DataFrame:

    summarized_data = data.groupby(datetime_col_name).count().iloc[:, 0].to_frame(name=summarize_feature_name)
    summarized_data.reset_index(inplace=True)
    summarized_data.sort_values(by=datetime_col_name, inplace=True, ascending=True)

    summarized_data["Cumsum"] = summarized_data[summarize_feature_name].cumsum()
    summarized_data["Daily Difference"] = summarized_data[summarize_feature_name].diff()
    summarized_data["Growth Factor"] = summarized_data["Daily Difference"] / summarized_data["Daily Difference"].shift(1)
    summarized_data["Weekly Rolling Average"] = summarized_data[summarize_feature_name].rolling(window=7).mean().round()
    summarized_data["Pct Change"] = summarized_data["Weekly Rolling Average"].pct_change()
    summarized_data["Weekly Average CumSum"] = summarized_data["Weekly Rolling Average"].cumsum()
    
    idx = summarized_data['Weekly Average CumSum'].sub(initial_number_of_cases).abs().idxmin()
    summarized_data["Epidemiological Days"]  = (summarized_data[datetime_col_name] - summarized_data.loc[idx, datetime_col_name]) / np.timedelta64(1, 'D')

    return summarized_data


@typechecked
def make_cases_training_data(datetime_col_name: str="notification_date"
                            ,outbreak_start_date: str="2021-06-01") -> \
                            Union[pd.DataFrame, int]:
    """Make training dataset

    Args:
        datetime_col_name (str.): Name of datetime column
        outbreak_start_date (str.): Start date of outbreak (ie. date after which cases sources started to be local transmission)

    Returns:
        (pd.DataFrame): Training dataset
    """

    # Get raw cases data
    raw_data = get_cases_data()
    raw_data = to_datetime(datetime_col_name, raw_data)

    # Subset data to the latest outbreak
    interim_data = subset_latest_outbreak(outbreak_start_date, 'Overseas', raw_data)

    # Aggregate number of cases by day
    return get_daily_cases_stats(interim_data)
    

@typechecked
def make_train_test_split(data: pd.DataFrame, target: str=None, 
                        strategy: str=None, train_size: float=0.75):
    """Make train_test_split

    Args:
        data (pd.DataFrame): Data set containing predictor and target
        target (str): Name of target column
        strategy (str, optional): Choice of train-test split. Defaults to None.
        train_size (float, optional): Percentage of data used for training. Defaults to 0.75.

    Returns:
        (np.ndarray): Train and test data and associated targets
    """
    mask = data['Epidemiological Days'] >= 0
    X = data.loc[mask, ["Epidemiological Days"]]
    y = data.loc[mask, target]

    if strategy == "time":
        test_size = 1.0 - train_size
        X_train = X.head(ceil(len(X)*train_size))
        X_test = X.tail(ceil(len(X)*test_size))
        y_train = y.head(ceil(len(y)*train_size))
        y_test = y.tail(floor(len(y)*test_size))

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    return X_train.values.squeeze(), X_test.values.squeeze() \
        ,y_train.values.squeeze(), y_test.values.squeeze()


@typechecked
def get_prior_distr_params(y_train: np.ndarray):
    """
    Get the prior distribution parameters for the training data

    Args:
        y_train (np.ndarray): Target training data

    Returns:
        (float): Statistics of target data distribution
    """

    initial_number_of_cases = np.min(y_train)
    daily_number_of_cases_std = np.std(y_train)
    pct_change = np.diff(y_train, axis=0) / y_train[:-1]
    average_pct_change = np.mean(pct_change)
    std_pct_change = np.std(pct_change)

    return initial_number_of_cases, daily_number_of_cases_std\
        ,average_pct_change, std_pct_change
