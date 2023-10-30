import pandas as pd
import numpy as np

def random_walk_forecast(train: pd.DataFrame, test: pd.DataFrame):
    last_value = train.values[-1]
    forecast_values = [last_value]

    for _ in range(len(test) - 1):
        step = np.random.normal()  # drawing from a standard normal distribution for random steps
        last_value += step
        forecast_values.append(last_value)

    # Convert the forecast_values to a DataFrame with the index from the test DataFrame
    forecast_df = pd.DataFrame(forecast_values, columns=['Forecast'], index=test.index)
    
    return forecast_df
