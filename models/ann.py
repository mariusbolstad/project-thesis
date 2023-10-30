from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Prepare the dataset
def create_ann_dataset(data, lookback, is_test=False, ):
    X, y = [], []
    if is_test:  # for test data, we just need the last entry for 1-step ahead forecast
        X.append(data[-lookback:])
        return np.array(X).reshape(1, lookback, 1), None
    else:
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

# Design the ANN model
def create_ann_model(lookback, hidden_units=2):
    model = Sequential()
    model.add(Dense(units=hidden_units, input_dim=lookback, activation='linear'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def ANN_forecast(train, test, steps_ahead, lookback=2, hidden_units=2):
    # Rescale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    
    X_train, y_train = create_ann_dataset(train_scaled, lookback)
    
    model = create_ann_model(lookback=lookback, hidden_units=hidden_units)
    model.fit(X_train, y_train, epochs=3, batch_size=1)


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


def ANN_diff_forecast(train, test, steps_ahead, lookback=2, hidden_units=2):
    # Difference the data
    diff_train = train.diff().dropna()
    
    # Rescale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(diff_train.values.reshape(-1, 1))
    
    X_train, y_train = create_ann_dataset(train_scaled, lookback)
    
    model = create_ann_model(lookback=lookback, hidden_units=hidden_units)
    model.fit(X_train, y_train, epochs=3, batch_size=1)

    scaled_fitted_values = model.predict(X_train)
    r2 = r2_score(y_train, scaled_fitted_values)
    # Inverse the scaling and then inverse the differencing for fitted values
    fitted_diff_values_array = scaler.inverse_transform(scaled_fitted_values)
    fitted_values_array = np.concatenate([train[:1].values, train[1:].values + fitted_diff_values_array.cumsum(axis=0)], axis=0)
    fitted_df = pd.DataFrame(fitted_values_array, columns=['Fitted Values'], index=train.index)
    
    # Starting with the last 'lookback' data points from the differenced training set
    input_data = train_scaled[-lookback:]
    
    # List to store forecasted points in the differenced form
    scaled_forecast = []

    # Iteratively predict steps ahead
    for _ in range(steps_ahead):
        X_test, _ = create_ann_dataset(input_data, lookback, is_test=True)
        prediction = model.predict(X_test)
        scaled_forecast.append(prediction[0][0])
        
        # Append the new prediction to the end of input_data and remove the oldest value
        input_data = np.append(input_data[1:], prediction)

    # Inverse the scaling for forecasted differenced values
    forecast_diff_array = scaler.inverse_transform(np.array(scaled_forecast).reshape(-1, 1))
    # Inverse the differencing to obtain forecast in original scale
    forecast_array = np.concatenate([train[-1:].values, train[-1:].values + forecast_diff_array.cumsum(axis=0)], axis=0)[1:]
    
    forecast_df = pd.DataFrame(forecast_array, columns=['Forecast'], index=test.index[:steps_ahead])
    
    return fitted_df, forecast_df
