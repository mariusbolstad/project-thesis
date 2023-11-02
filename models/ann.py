from keras.models import Sequential
from keras.layers import Dense
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.callbacks import EarlyStopping



# Prepare the dataset
def create_ann_dataset(data, lookback, is_test=False, exog=None):
    X, y = [], []
    if exog is not None:
        data = np.hstack((data, exog))
    if is_test:
        X.append(data[-lookback:].flatten())
        return np.array(X), None
    else:
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i].flatten())
            y.append(data[i][0])
        return np.array(X), np.array(y)

# Design the ANN model
def create_ann_model(lookback, num_features=1, hidden_units=2, activation="sigmoid"):
    model = Sequential()
    model.add(Dense(units=hidden_units, input_dim=lookback * num_features, activation=activation))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def ANN_forecast(train, test, steps_ahead, lookback=2, hidden_units=2):
    # Rescale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    
    X_train, y_train = create_ann_dataset(train_scaled, lookback)
    
    model = create_ann_model(lookback=lookback, hidden_units=hidden_units)
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)

    model.fit(X_train, y_train, epochs=3, batch_size=1, callbacks=[early_stop])


    scaled_fitted_values = model.predict(X_train)
    r2 = r2_score(y_train, scaled_fitted_values)
    #print("R2: " + str(r2))
    fitted_values_array = scaler.inverse_transform(scaled_fitted_values)
    fitted_df = pd.DataFrame(np.concatenate((train.values[:lookback].reshape(-1, 1), fitted_values_array), axis=0), columns=['Fitted Values'], index=train.index)
     # Starting with the last 'lookback' data points from the training set
    input_data = train_scaled[-lookback:]
    
    # List to store forecasted points
    scaled_forecast = []

    # Iteratively predict steps ahead
    for _ in range(steps_ahead):
        X_test, _ = create_ann_dataset(input_data, lookback, is_test=True)
        prediction = model.predict(X_test)
        scaled_forecast.append(prediction[0][0])  # Add the new prediction
        
        # Append the new prediction to the end of input_data and remove the oldest value
        input_data = np.append(input_data[1:], prediction)
    # Inverse the scaling on the forecasted values
    forecast_array = scaler.inverse_transform(np.array(scaled_forecast).reshape(-1, 1))
    forecast_df = pd.DataFrame(forecast_array, columns=['Forecast'], index=test.index[:steps_ahead])
    
    return fitted_df, forecast_df

def create_scaler(scaler_key):
    if scaler_key == "robust":
        return RobustScaler()
    elif scaler_key == "standard":
        return StandardScaler()
    return RobustScaler


def ANN_diff_forecast(train, 
                      test, 
                      steps_ahead, 
                      lookback=2, 
                      hidden_units=2, 
                      epochs=10, 
                      exog_train=None, 
                      exog_test=None,
                      ann_diff=True,
                      exog_diff=True,
                      scaler="robust",
                      activation="sigmoid"):
    # Rescale and Difference the data
    if ann_diff:
        train_scaler = create_scaler(scaler_key=scaler)
        diff_train = train.diff().dropna()
        train_scaled = train_scaler.fit_transform(diff_train.values.reshape(-1, 1))
    else:
        train_scaler_2 = create_scaler(scaler_key=scaler)
        train_scaled = train_scaler_2.fit_transform(train.values.reshape(-1, 1))

    # If exog variables are provided, scale them and concatenate
    if exog_train is not None:
        if exog_diff:
            diff_exog_train = exog_train.diff().dropna()
        elif ann_diff and not exog_diff:
            diff_exog_train = exog_train.iloc[1:, :]
        exog_scaler = create_scaler(scaler_key=scaler)
        # Scale the first column
        first_col = diff_exog_train.iloc[:, 0]
        exog_train_scaled = exog_scaler.fit_transform(first_col.values.reshape(-1, 1))
        # If there's more than one column, loop through the remaining columns and scale them
        if diff_exog_train.shape[1] > 1:
            for i in range(1, diff_exog_train.shape[1]):
                exog_scaler_1 = create_scaler(scaler_key=scaler)
                col = diff_exog_train.iloc[:, i]
                scaled_col = exog_scaler_1.fit_transform(col.values.reshape(-1, 1))
                exog_train_scaled = np.column_stack((exog_train_scaled, scaled_col))
        train_data_combined = np.hstack((train_scaled, exog_train_scaled))
        X_train, y_train = create_ann_dataset(train_scaled, lookback, exog=exog_train_scaled)
    else:
        train_data_combined = train_scaled
        X_train, y_train = create_ann_dataset(train_scaled, lookback)

    
    num_features = train_data_combined.shape[1]

    model = create_ann_model(lookback=lookback, hidden_units=hidden_units, num_features=num_features, activation=activation)
    model.fit(X_train, y_train, epochs=epochs, batch_size=1)

    scaled_fitted_values = model.predict(X_train)
    r2 = r2_score(y_train, scaled_fitted_values)
    print("R2: " + str(r2))
    if ann_diff:
        # Inverse the scaling and then inverse the differencing for fitted values
        fitted_diff_values_array = train_scaler.inverse_transform(scaled_fitted_values)
        t1 = train[:1 + lookback].values.reshape(-1, 1)
        t2 = train[1 + lookback:].values.reshape(-1, 1) + fitted_diff_values_array.cumsum(axis=0)
        fitted_values_array = np.concatenate([t1, t2], axis=0)
    else:
        fitted_values_array = train_scaler.inverse_transform(scaled_fitted_values)

    fitted_df = pd.DataFrame(fitted_values_array, columns=['Fitted Values'], index=train.index)
    
    # Starting with the last 'lookback' data points from the differenced training set
    input_data = train_data_combined[-lookback:]
    
    # List to store forecasted points in the differenced form
    scaled_forecast = []

    # Iteratively predict steps ahead
    for _ in range(steps_ahead): # Append exog_test data if it exists
        X_test, _ = create_ann_dataset(input_data, lookback, is_test=True)
        prediction = model.predict(X_test)
        scaled_forecast.append(prediction[0][0])

        
        # Update the input data for next iteration
        input_data[-1, 0] = prediction

    # Inverse the scaling for forecasted differenced values
    forecast_diff_array = train_scaler.inverse_transform(np.array(scaled_forecast).reshape(-1, 1))
    # Inverse the differencing to obtain forecast in original scale
    # Extract the last value of the train set

    # Update forecast_array to start with the last value of train and add the cumulative sum of the differences
    last_value = train.iloc[-1]  # This will give a scalar

    forecast_array = last_value + forecast_diff_array.cumsum(axis=0)

    # Create the DataFrame
    forecast_df = pd.DataFrame(forecast_array, columns=['Forecast'], index=test.index[:steps_ahead])

    
    return fitted_df, forecast_df




