import pandas as pd
import numpy as np

def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size][::-1])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def prepare_time_series_data(data, window_size):
    data = data.sort_values(['Location ID', 'Timestamp'])
    grouped_data = data.groupby('Location ID')

    X, y = [], []
    for _, group in grouped_data:
        kwh_data = group['kwh/day'].values
        X_group, y_group = create_sliding_window(kwh_data, window_size)
        X.extend(X_group)
        y.extend(y_group)

    return np.array(X), np.array(y), grouped_data

data = pd.DataFrame({
    'Location ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    'Timestamp': pd.date_range('2023-01-01', periods=5).tolist() * 2,
    'kwh/day': [100, 110, 95, 105, 120, 150, 140, 160, 130, 155],
    'Population': [5000, 5100, 5150, 5200, 5250, 3000, 3100, 3200, 3300, 3350],
    'Avg Income': [60000, 60500, 61000, 61200, 61500, 80000, 80500, 81000, 81500, 82000]
})
window_size = 2
X, y, grouped_data = prepare_time_series_data(data, window_size)

processed_data = pd.DataFrame(columns=['Location ID', 'Target (t+1)', 'Feature 1 (t)', 'Feature 2 (t-1)', 'Population', 'Avg Income'])

idx = 0
for loc_id, group in grouped_data:
    for i in range(group.shape[0] - window_size):
        extra_features = group.iloc[i + window_size - 1][['Population', 'Avg Income']].values
        processed_data.loc[idx] = [loc_id, y[idx], X[idx][0], X[idx][1], *extra_features]
        idx += 1

# Print the processed data
print(data)
# show all columns of processed data frame
pd.set_option('display.max_columns', None)
print(processed_data)