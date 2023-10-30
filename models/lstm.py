import numpy as np
import pandas as pd
from sklearn.base import r2_score
from sklearn.preprocessing import MinMaxScaler
import ann

def ANN_forecast(train, test, steps_ahead, lookback=2, hidden_units=2):
    # Rescale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    
    X_train, y_train = ann.create_ann_dataset(train_scaled, lookback)
    
    model = ann.create_ann_model(lookback=lookback, hidden_units=hidden_units)
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
        X_test, _ = ann.create_ann_dataset(input_data, lookback, is_test=True)
        prediction = model.predict(X_test)
        scaled_forecast.append(prediction[0][0])  # Add the new prediction
        
        # Append the new prediction to the end of input_data and remove the oldest value
        input_data = np.append(input_data[1:], prediction)
    # Inverse the scaling on the forecasted values
    forecast_array = scaler.inverse_transform(np.array(scaled_forecast).reshape(-1, 1))
    forecast_df = pd.DataFrame(forecast_array, columns=['Forecast'], index=test.index[:steps_ahead])
    
    return fitted_df, forecast_df