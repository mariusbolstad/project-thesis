import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pprint
from datetime import timedelta
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from arch.unitroot import PhillipsPerron



def ARIMA_forecast(train: pd.DataFrame,
                   test: pd.DataFrame,
                   steps_ahead, 
                   p: int=3, 
                   i: int=1, 
                   q: int=3,
                   plot: bool=False,
                   exog: pd.DataFrame=None):

    # Use the exog data corresponding to the train data
    exog_train = None
    if exog is not None:
        exog_train = exog.loc[train.index]
    
    arima_model = ARIMA(train.dropna(), order=(p, i, q), exog=exog_train)
    model = arima_model.fit()
    print(model.summary())
    
    # Extract fitted values and create a DataFrame
    fitted_df = pd.DataFrame(model.fittedvalues, columns=['Fitted'], index=train.index)

    # Use the exog data corresponding to the test (forecast) data
    exog_test = None
    if exog is not None:
        exog_test = exog.loc[test.index][:steps_ahead]

    forecast = model.get_forecast(steps=steps_ahead, exog=exog_test)
    mean_forecast = forecast.predicted_mean

    # Convert mean_forecast to a DataFrame with the index from the test DataFrame
    forecast_df = pd.DataFrame(mean_forecast.values, columns=['Forecast'], index=test.index[:steps_ahead])


    if plot:
        # Sample plot code (can be extended or modified based on requirements)
        #train.plot(label='Training Data')
        #fitted_df.plot(label='Fitted Values')
        #forecast_df.plot(label='Forecast')
        #plt.legend()
        #plt.show()
        
        # Plotting the residuals
        plt.figure(figsize=(12, 6))
        residuals = pd.DataFrame(model.resid)
        residuals.plot(title="Residuals")
        plt.legend()
        plt.title("ARIMA Model Residuals")
        plt.show()
        plot_acf(residuals, lags=40)
        plt.title('ACF of ARIMA Model Residuals')
        plt.show()
        

    return fitted_df, forecast_df, model


def ARIMAX_forecast(train: pd.DataFrame,
                   test: pd.DataFrame,
                   steps_ahead, 
                   p: int=3, 
                   i: int=1, 
                   q: int=3,
                   plot: bool=False):

    arima_model = ARIMA(train.dropna(), order=(p, i, q))
    model = arima_model.fit()
    #print(model.summary())

    # Extract fitted values and create a DataFrame
    fitted_df = pd.DataFrame(model.fittedvalues, columns=['Fitted'], index=train.index)

    forecast = model.get_forecast(steps=steps_ahead)
    mean_forecast = forecast.predicted_mean

    # Convert mean_forecast to a DataFrame with the index from the test DataFrame
    forecast_df = pd.DataFrame(mean_forecast.values, columns=['Forecast'], index=test.index)



# not currently using
def ARIMA_forecasts(data: pd.DataFrame,
                   steps_ahead: int=1, 
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
            plt.plot(true_values.iloc[:train_size + steps_ahead].index, true_values.iloc[:train_size + steps_ahead].values, label='observed')
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
        test = data["CLOSE"].iloc[train_size:train_size + steps_ahead]
        #print(train.head())

        arima_model = ARIMA(train.dropna(), order=(p, i, q))
        model = arima_model.fit()
        #print(model.fittedvalues.head())

        forecast = model.get_forecast(steps=steps_ahead)
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
        train_size += steps_ahead  # Shift the training data for next forecast
        
        if plot:
            plot_forecast()
            

    return forecast_wrappers

def stationarity_tests(df):
    # STATIONARITY TESTS

    # augmented dickey fuller

    values = df["CLOSE"]

    result = adfuller(values.dropna())
    print('p-value: ', result[1])

    result = adfuller(values.diff().dropna())
    print('p-value: ', result[1])

    result = adfuller(values.diff().diff().dropna())
    print('p-value: ', result[1])

    # Phillips-Perron

    pp_test = PhillipsPerron(values.dropna())
    print(pp_test.summary())


    pp_test = PhillipsPerron(values.diff().dropna())
    print(pp_test.summary())

    # KPSS test
    statistic, p_value, n_lags, critical_values = kpss(values, regression='c')

    print(f"KPSS Statistic: {statistic}")
    print(f"P-value: {p_value}")
    print(f"Num Lags: {n_lags}")
    print("Critical Values:", critical_values)

    if p_value < 0.05:
        print("Series is not stationary")
    else:
        print("Series is stationary")

    statistic, p_value, n_lags, critical_values = kpss(values.diff().dropna(), regression='c')

    print(f"KPSS Statistic: {statistic}")
    print(f"P-value: {p_value}")
    print(f"Num Lags: {n_lags}")
    print("Critical Values:", critical_values)

    if p_value < 0.05:
        print("Series is not stationary")
    else:
        print("Series is stationary")

def find_arima_spec(df):
    #df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    #print("Shape of data", df.shape)
    #print(df.head())
    df["CLOSE"].plot(figsize=(12,5))
    #stationarity_tests(df)
    #plt.show()
    stepwise_fit = auto_arima(df["CLOSE"].dropna(), trace=True, suppress_warnings=True, n_fits=50)
    return stepwise_fit.summary()
    
    
    
    
    
    
