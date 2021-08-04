import pandas as pd
import numpy as np

def get_cases_data() -> pd.DataFrame:
    """Get cases data from csv file

    Returns:
        (pd.DataFrame): Cases data
    """
    return pd.read_csv('https://data.nsw.gov.au/data/dataset/97ea2424-abaf-4f3e-a9f2-b5c883f42b6a/resource/2776dbb8-f807-4fb2-b1ed-184a6fc2c8aa/download/confirmed_cases_table4_location_likely_source.csv')


def to_datetime(column: str, data: pd.DataFrame)  -> pd.DataFrame:
    """Convert date column to datetime

    Returns:
        (pd.DataFrame): Cases data
    """

    data[column] = pd.to_datetime(data[column])

    return data


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


def get_daily_cases_stats(data: pd.DataFrame) -> pd.DataFrame:

    cases_agg = data.groupby("notification_date").count().iloc[:, 0].to_frame(name="Daily Number of Cases")
    cases_agg.reset_index(inplace=True)
    cases_agg.sort_values(by="notification_date", inplace=True, ascending=True)

    cases_agg["Pct Change"] = cases_agg["Daily Number of Cases"].pct_change()
    cases_agg["Cumsum"] = cases_agg["Daily Number of Cases"].cumsum()
    cases_agg["Daily Difference"] = cases_agg["Daily Number of Cases"].diff()
    cases_agg["Growth Factor"] = cases_agg["Daily Difference"] / cases_agg["Daily Difference"].shift(1)
    cases_agg["Weekly Rolling Average"] = cases_agg["Daily Number of Cases"].rolling(window=7).mean().round()
    cases_agg["Weekly Average CumSum"] = cases_agg["Weekly Rolling Average"].cumsum()
    
    idx = cases_agg['Weekly Rolling Average'].sub(50).abs().idxmin()
    cases_agg["Epidemiological Days"]  = (cases_agg["notification_date"] - cases_agg.loc[idx, "notification_date"]) / np.timedelta64(1, 'D')

    return cases_agg