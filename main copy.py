from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from models.arima_forecasts import ARIMA_forecast, find_arima_spec
from models.random_walk import random_walk_forecast
from models.ann import ANN_forecast
from helper_functions import compute_rmse, load_monthly_baci_data, load_daily_baci_data, load_weekly_baci_data, load_daily_exog_data, load_weekly_exog_data, load_monthly_exog_data, load_monthly_exog_spot_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from models.ann import ANN_diff_forecast
import csv
import os

# Suppress all warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ValueWarning)

solstorm = "/storage/users/mariumbo/log3.csv"
local = "log3.csv"
def log_print(data_dict):
    # Check if the CSV file already exists to decide whether to write headers
    file_exists = os.path.isfile(local)

    with open(local, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data_dict)
        
        
    
        
def ARIMA_vs_RW_vs_ANN(data, 
                       train_ratio=0.8, 
                       steps_ahead=1, 
                       n_forecasts=1, 
                       plot=False, 
                       res=15, 
                       arima_spec=(1, 1, 2), 
                       ann_spec=(2, 2),
                       ann_diff=True,
                       exog_diff=True,
                       activation="sigmoid",
                       scaler="robust",
                       epochs=10, 
                       plot_all=True,
                       arima_f=True,
                       ann_f=True,
                       comb_f=True,
                       arimax_f=True,
                       annx_f=True,
                       combx_f=True,
                       exog=None,
                       log=False):


    def plot_forecast(arima_fitted_df, 
                      arima_forecast_df, 
                      rw_forecast_df,
                      ann_fitted_df,
                      ann_forecast_df,
                      train_df,
                      test_df,
                      res=15):
        true_values = pd.concat([train, test])

        # Plot
        plt.figure(figsize=(15,7))

        plt.plot(true_values.index[-res - steps_ahead:], true_values[-res - steps_ahead:].values, color="blue", label='observed')
        plt.plot(arima_fitted_df.index[-res:], arima_fitted_df[-res:].values, color='green', label='ARIMA fitted')
        plt.plot(ann_fitted_df.index[-res:], ann_fitted_df[-res:].values, color='red', label='ANN fitted')
        
        
        # Rename columns of train_df.iloc[-1] to match rw_forecast_df's columns
        # Convert the last value of train_df to a dataframe with the same column name as rw_forecast_df
        renamed_row = pd.DataFrame([arima_fitted_df.values[-1]], columns=arima_forecast_df.columns, index=[arima_fitted_df.index[-1]])
        arima_forecast_df_merged = pd.concat([renamed_row, arima_forecast_df])
    
        renamed_row = pd.DataFrame([ann_fitted_df.values[-1]], columns=ann_forecast_df.columns, index=[ann_fitted_df.index[-1]])
        ann_forecast_df_merged = pd.concat([renamed_row, ann_forecast_df])
      
        renamed_row = pd.DataFrame([train_df.iloc[-1]], columns=rw_forecast_df.columns, index=[train_df.index[-1]])
        rw_forecast_df_merged = pd.concat([renamed_row, rw_forecast_df])


        plt.plot(arima_forecast_df_merged.index, arima_forecast_df_merged.values, color='green', label='ARIMA forecast', linestyle=":")
        plt.plot(rw_forecast_df_merged.index, rw_forecast_df_merged.values, color="purple", label="RW forecast", linestyle=":")
        plt.plot(ann_forecast_df_merged.index, ann_forecast_df_merged.values, color="red", label="ANN forecast", linestyle=":")

        plt.legend()
        plt.show()
        print(f"ARIMA MSE: {compute_rmse(test, arima_forecast_df)}")
        print(f"RW MSE: {compute_rmse(test, rw_forecast_df)}")
        print(f"ANN MSE: {compute_rmse(test, ann_forecast_df)}")




    train_size = int(train_ratio * len(data))
    arima_rmses, arimax_rmses, rw_rmses, ann_rmses, annx_rmses, comb_rmses, combx_rmses = [], [], [], [], [], [], []
    
    # Initialize empty DataFrames with 'Date' as the index
    columns = ['Forecast']
    first_train_size = train_size
    arima_all_forecasts_df = arimax_all_forecasts_df = rw_all_forecasts_df = ann_all_forecasts_df = annx_all_forecasts_df = comb_all_forecasts_df =combx_all_forecasts_df = pd.DataFrame(columns=columns).set_index(pd.DatetimeIndex([], name='Date'))
    epsilon = 1
    if log:
        for i in range(len(data.columns)):
            data.iloc[:, i] = np.maximum(data.iloc[:, i], 0)
        for i in range(len(exog.columns)):
            exog.iloc[:, i] = np.maximum(exog.iloc[:, i], 0)
        data = data.apply(lambda x: np.log(x+epsilon))
        exog = exog.apply(lambda x: np.log(x+epsilon))
    
    for _ in range(n_forecasts):
   
        print("Forecast number " + str(_ + 1))
        train = data["CLOSE"].iloc[:train_size]
        test = data["CLOSE"].iloc[train_size:train_size + steps_ahead]
        exog_train = exog.iloc[:train_size]
        exog_test = exog.iloc[train_size: train_size + steps_ahead]

        # Random Walk forecast
        rw_forecast_df = random_walk_forecast(train, test)
        rw_rmse = compute_rmse(test, rw_forecast_df)
        rw_rmses.append(rw_rmse)
        rw_all_forecasts_df = pd.concat([rw_all_forecasts_df, rw_forecast_df])


        if arima_f:
            # ARIMA forecast
            arima_fitted_df, arima_forecast_df, arima_model = ARIMA_forecast(train, 
                                                                            test, 
                                                                            steps_ahead, 
                                                                            p=arima_spec[0], 
                                                                            i=arima_spec[1], 
                                                                            q=arima_spec[2],
                                                                            plot=False,
                                                                            exog=None)
            arima_rmse = compute_rmse(test, arima_forecast_df)
            arima_rmses.append(arima_rmse)
            arima_all_forecasts_df = pd.concat([arima_all_forecasts_df, arima_forecast_df])
        if arimax_f:
            # ARIMAX forecast
            arimax_fitted_df, arimax_forecast_df, arimax_model = ARIMA_forecast(train, 
                                                                            test, 
                                                                            steps_ahead, 
                                                                            p=arima_spec[0], 
                                                                            i=arima_spec[1], 
                                                                            q=arima_spec[2],
                                                                            plot=False,
                                                                            exog=exog)
            #print(arima_forecast_df.values)
            arimax_rmse = compute_rmse(test, arimax_forecast_df)
            arimax_rmses.append(arimax_rmse)
            arimax_all_forecasts_df = pd.concat([arimax_all_forecasts_df, arimax_forecast_df])


        if ann_f:
            # ANN forecast
            ann_fitted_df, ann_forecast_df = ANN_diff_forecast(train=train, 
                                                        test=test, 
                                                        steps_ahead=steps_ahead,
                                                        lookback=ann_spec[0],
                                                        hidden_units=ann_spec[1],
                                                        activation=activation,
                                                        ann_diff=ann_diff,
                                                        scaler=scaler,
                                                        epochs=epochs)
            ann_rmse = compute_rmse(test, ann_forecast_df)
            ann_rmses.append(ann_rmse)
            ann_all_forecasts_df = pd.concat([ann_all_forecasts_df, ann_forecast_df])
        
        if annx_f:
             # ANNX forecast
            annx_fitted_df, annx_forecast_df = ANN_diff_forecast(train=train, 
                                                        test=test, 
                                                        steps_ahead=steps_ahead,
                                                        lookback=ann_spec[0],
                                                        hidden_units=ann_spec[1],
                                                        exog_train=exog_train,
                                                        exog_test=exog_test,
                                                        epochs=epochs,
                                                        activation=activation,
                                                        ann_diff=ann_diff,
                                                        exog_diff=exog_diff,
                                                        scaler=scaler,)
            annx_rmse = compute_rmse(test, annx_forecast_df)
            annx_rmses.append(annx_rmse)
            annx_all_forecasts_df = pd.concat([annx_all_forecasts_df, annx_forecast_df])

        if comb_f:
            # Combination forecast
            comb_forecast_df = pd.DataFrame()
            comb_forecast_df["Forecast"] = (ann_forecast_df["Forecast"] + arima_forecast_df["Forecast"]) / 2
            comb_rmse = compute_rmse(test, comb_forecast_df)

            comb_rmses.append(comb_rmse)
            comb_all_forecasts_df = pd.concat([comb_all_forecasts_df, comb_forecast_df])
            
        if combx_f:
            # Combination forecast
            combx_forecast_df = pd.DataFrame()
            combx_forecast_df["Forecast"] = (annx_forecast_df["Forecast"] + arimax_forecast_df["Forecast"]) / 2
            combx_rmse = compute_rmse(test, combx_forecast_df)

            combx_rmses.append(combx_rmse)
            combx_all_forecasts_df = pd.concat([combx_all_forecasts_df, combx_forecast_df])
        

        train_size += steps_ahead

        if plot:
            plot_forecast(arima_fitted_df=arima_fitted_df,
                          arima_forecast_df=arima_forecast_df,
                          rw_forecast_df=rw_forecast_df,
                          ann_fitted_df=ann_fitted_df,
                          ann_forecast_df=ann_forecast_df,
                          train_df=train,
                          test_df=test,
                          res=res)
    if plot_all:
        plt.gca().set_prop_cycle(None)  # Reset color cycle
        plt.plot(data.index[first_train_size:train_size + steps_ahead - 1], data[first_train_size:train_size + steps_ahead - 1].values, color="blue", label='observed')
        plt.gca().set_prop_cycle(None)  # Reset color cycle
        plt.plot(rw_all_forecasts_df.index, rw_all_forecasts_df.values, color='purple', label='RW forecast', linestyle=":")

        if arima_f:
            plt.plot(arima_all_forecasts_df.index, arima_all_forecasts_df.values, color='green', label='ARIMA forecast', linestyle=":")
        if ann_f:
            plt.plot(ann_all_forecasts_df.index, ann_all_forecasts_df.values, color='red', label='ANN forecast', linestyle=":")
        if comb_f:
            plt.plot(comb_all_forecasts_df.index, comb_all_forecasts_df.values, color='black', label='ANN/ARIMA combination', linestyle=":")
        if arimax_f:
            plt.plot(arimax_all_forecasts_df.index, arimax_all_forecasts_df.values, color='brown', label='ARIMAX forecast', linestyle=":")
        if annx_f:
            plt.plot(annx_all_forecasts_df.index, annx_all_forecasts_df.values, color='pink', label='ANNX forecast', linestyle=":")


        # Rotate x-axis labels for better clarity
        plt.xticks(rotation=45)

        # Optional: Use fewer date ticks on the x-axis
        #import matplotlib.dates as mdates
        #plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Every week

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())  # Display unique labels only

        plt.tight_layout()  # Adjust the layout for better display
        plt.show()

        


    return arima_rmses, rw_rmses, ann_rmses, comb_rmses, arimax_rmses, annx_rmses, combx_rmses



#find_arima_spec(df)
#print(exog.head())
#print(df.head())



for i in range(2):
    for j in range(2):
        if j == 0:
            data_freq = "weekly"
            arima_spec = (3, 1, 1)
        else:
            data_freq = "daily"
            arima_spec = (3, 1, 3)
        exog_inc = True
        if data_freq == "daily":
            df = load_daily_baci_data()
            if exog_inc:
                exog = load_daily_exog_data()
        elif data_freq == "weekly":
            df = load_weekly_baci_data()
            if exog_inc:
                exog = load_weekly_exog_data()
        else:
            df = load_monthly_baci_data()
            if exog_inc:
                exog = load_monthly_exog_spot_data()

        if exog_inc:
            # Inner join both dataframes on their index to ensure they have the same timestamp
            result = df.join(exog, how='inner', lsuffix='_baci', rsuffix='_ironfut')

            # Split the result back into individual dataframes if needed
            df = result[['CLOSE']]
            exog = result[['IRON_CLOSE', "COAL_CLOSE"]]

            # Rename the columns of the individual dataframes to 'CLOSE'
            df = df.rename(columns={'CLOSE_baci': 'CLOSE'})
        
        
        train_ratio = 0.8
        n_forecasts = 30
        steps_ahead = 1
        res = 200
        ann_spec = (4, 2)
        epochs = 20
        activation = "sigmoid"
        exog_diff = False
        ann_diff = True
        scaler = "standard"
        plot = False
        plot_all = False
        res = 50
        exog = exog
        if i == 1:
            log = False
        else:
            log = True
        arima_f, arimax_f, ann_f, annx_f, comb_f, combx_f = True, True, True, True, True, True
        arima_rmses, rw_rmses, ann_rmses, comb_rmses, arimax_rmses, annx_rmses, combx_rmses = ARIMA_vs_RW_vs_ANN(data=df, 
                                                                        res=res, 
                                                                        train_ratio=train_ratio,
                                                                        n_forecasts=n_forecasts,
                                                                        steps_ahead=steps_ahead,
                                                                        plot=False,
                                                                        plot_all=plot_all,
                                                                        arima_spec=arima_spec,
                                                                        ann_spec=ann_spec,
                                                                        epochs=epochs,
                                                                        activation=activation,
                                                                        ann_diff=ann_diff,
                                                                        exog_diff=exog_diff,
                                                                        scaler=scaler,
                                                                        exog=exog,
                                                                        arima_f=arima_f,
                                                                        arimax_f = arimax_f,
                                                                        ann_f=ann_f,
                                                                        annx_f=annx_f,
                                                                        comb_f=comb_f,
                                                                        combx_f=combx_f,
                                                                        log=log
                                                                        )


        # Prepare the data dictionary with all parameters and results
        log_data = {
            "Data frequency": data_freq,
            "Train Ratio": train_ratio,
            "Number of Forecasts": n_forecasts,
            "Steps Ahead": steps_ahead,
            "Resolution": res,
            "ARIMA Specification": str(arima_spec),
            "ANN Specification": str(ann_spec),
            "Epochs": epochs,
            "Activation Function": activation,
            "ANN Differences": ann_diff,
            "Exogenous Differences": exog_diff,
            "Scaler": scaler,
            "Log": log, 
            "Plot": plot,
            "Plot All": plot_all,
            "Exogenous Variable": str(exog.columns.tolist()),  # Assuming exog is a DataFrame
            "ARIMA RMSEs": 'N/A' if not arima_rmses else round(np.average(arima_rmses), 2),
            "ARIMAX RMSEs": 'N/A' if not arimax_rmses else round(np.average(arimax_rmses), 2),
            "Random Walk RMSEs": 'N/A' if not rw_rmses else round(np.average(rw_rmses), 2),
            "ANN RMSEs": 'N/A' if not ann_rmses else round(np.average(ann_rmses), 2),
            "ANNX RMSEs": 'N/A' if not annx_rmses else round(np.average(annx_rmses), 2),
            "Combination RMSEs": 'N/A' if not comb_rmses else round(np.average(comb_rmses), 2),
            "CombinationX RMSEs": 'N/A' if not combx_rmses else round(np.average(combx_rmses), 2)

        }

        # Call log_print to write the data to the CSV file
        log_print(log_data)

