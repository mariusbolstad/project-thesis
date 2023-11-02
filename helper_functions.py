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


def load_daily_exog_data():
    df = pd.read_csv("./data/coal_futures_daily.csv")
    df2 = pd.read_csv("./data/iron_ore_futures_daily.csv")
    df["IRON_CLOSE"] = df2["CLOSE"]
    df = df.rename(columns={"CLOSE": "COAL_CLOSE"})
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def load_weekly_exog_data():
    df = pd.read_csv("./data/coal_futures_weekly.csv")
    df2 = pd.read_csv("./data/iron_ore_futures_weekly.csv")
    df["IRON_CLOSE"] = df2["CLOSE"]
    df = df.rename(columns={"CLOSE": "COAL_CLOSE"})
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def load_monthly_exog_data():
    df = pd.read_csv("./data/coal_futures_monthly.csv")
    df2 = pd.read_csv("./data/iron_ore_futures_monthly.csv")
    df["IRON_CLOSE"] = df2["CLOSE"]
    df = df.rename(columns={"CLOSE": "COAL_CLOSE"})
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def load_monthly_exog_spot_data():
    df = pd.read_csv("./data/coal_spot_monthly.csv")
    df2 = pd.read_csv("./data/iron_ore_spot_monthly.csv")
    df["IRON_CLOSE"] = df2["CLOSE"]
    df = df.rename(columns={"CLOSE": "COAL_CLOSE"})
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


