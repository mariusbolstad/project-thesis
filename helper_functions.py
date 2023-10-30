from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def compute_rmse(true_values, forecast):
    return np.sqrt(mean_squared_error(true_values, forecast))


def load_monthly_baci_data():
    df = pd.read_csv("./data/baci_monthly.csv")
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def load_daily_baci_data():
    df = pd.read_csv("./data/all_baci.csv")
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def load_weekly_baci_data():
    df = pd.read_csv("./data/baci_weekly.csv")
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


