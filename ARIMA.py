import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def ARIMA_forecast(train: pd.DataFrame,
                   test: pd.DataFrame,
                   months_ahead, 
                   p: int=3, 
                   i: int=1, 
                   q: int=3,
                   plot: bool=False):

    arima_model = ARIMA(train.dropna(), order=(p, i, q))
    model = arima_model.fit()
    forecast = model.get_forecast(steps=months_ahead)
    mean_forecast = forecast.predicted_mean
    return mean_forecast
    

def ARIMA_forecasts(data: pd.DataFrame,
                   months_ahead: int=1, 
                   train_ratio: float=0.8,
                   p: int=3, 
                   i: int=1, 
                   q: int=3,
                   n_forecasts: int=1,
                   plot: bool=True):
    
    def plot_forecast():
            true_values = data["CLOSE"]
            fitted_values = model.fittedvalues.shift(-1).dropna()
            # Append the last fitted value to the forecasted values
            forecast_values = np.insert(mean_forecast.values, 0, fitted_values.values[-1])
            forecast_dates = pd.date_range(start=fitted_values.index[-1], periods=len(forecast_values), freq='M')

            # Plot
            # Get standard error from the model fit
            std_error = np.std(model.resid)

            # Plot
            plt.figure(figsize=(15,7))
            plt.plot(true_values.iloc[:train_size + months_ahead].index, true_values.iloc[:train_size + months_ahead].values, label='observed')
            plt.plot(fitted_values.index, fitted_values.values, color='green', label='fitted')

            # Combined forecast values including the last fitted value
            forecast_values = np.insert(mean_forecast.values, 0, fitted_values.values[-1])
            forecast_dates = pd.date_range(start=fitted_values.index[-1], periods=len(forecast_values), freq='M')
            plt.plot(forecast_dates, forecast_values, color='red', label='forecast')

            if conf_int is not None:
                # Create confidence intervals for the last fitted value
                conf_int_last_fitted = [
                    fitted_values.values[-1] - 1.96 * std_error,  # lower bound
                    fitted_values.values[-1] + 1.96 * std_error   # upper bound
                ]

                # Append these intervals to the forecasted confidence intervals
                conf_int_values = np.vstack((
                    conf_int_last_fitted,
                    conf_int.values
                ))

                # Fill between for combined confidence intervals
                plt.fill_between(forecast_dates, conf_int_values[:, 0], conf_int_values[:, 1], color='pink')

            plt.legend()
            plt.show()

    forecast_wrappers = []
    
    train_size = int(train_ratio * len(data))
    for _ in range(n_forecasts):
        train = data["CLOSE"].iloc[:train_size]
        test = data["CLOSE"].iloc[train_size:train_size + months_ahead]
        print(train.head())

        arima_model = ARIMA(train.dropna(), order=(p, i, q))
        model = arima_model.fit()
        print(model.fittedvalues.head())

        forecast = model.get_forecast(steps=months_ahead)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()

        if not test.empty:  # Only compute RMSE if there is test data
            rmse = np.sqrt(mean_squared_error(test, mean_forecast))
        
        forecast_wrapper = {
            "mean_forecast": mean_forecast,
            "true_values": test,
            "conf_int": conf_int,
            "RMSE": rmse,
            "forecast_obj": forecast
        }

        forecast_wrappers.append(forecast_wrapper)
        train_size += months_ahead  # Shift the training data for next forecast
        
        if plot:
            plot_forecast()
            

    return forecast_wrappers
