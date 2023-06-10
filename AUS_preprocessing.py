import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from scipy.spatial import distance # for calculating distance between locations

# Define global variables

# These features are replaced by the "mean" of the features within the location-year-month group in the case of missing values.
features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm',
            'Humidity9am', 'Humidity3pm', 'Temp9am', 'Temp3pm', 'Pressure9am', 'Pressure3pm', 'WindGustSpeed']

# These features are those that include cases where all feature values within each group (location-year-month) are missing values. 
# A second filtration is required for classification.
features_2nd = ['Pressure9am', 'Pressure3pm', 'WindGustSpeed']

# These features are replaced by the "mode" of the features within the location-year-month group in the case of missing values.
categorical_features = ["WindGustDir", "WindDir9am", "WindDir3pm"]

# Special case handling for final filtration, from(features_2nd = ['Pressure9am', 'Pressure3pm', 'WindGustSpeed'])
special_cases = [("Newcastle", 2008, [12]), ("Penrith", 2008, [12]), 
                 ("Sydney", 2008, list(range(2, 13))), ("Albany", 2008, [12]), 
                 ("Albany", 2012, [8, 9])]

# Special case handling for final filtration, from categorical_features
exception_cases = [('Sydney', 2008, range(1, 13)), 
                   ('Newcastle', 2008, [12]), 
                   ('Albany', 2008, [12]), 
                   ('Albany', 2012, [8, 9])]



# function that loads the dataset and preprocess it 
def load_and_preprocess_data():
    data = pd.read_csv('weatherAUS.csv')
    data = data.drop(columns=['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']) # drop features that has abundant 'NaN'
    data = data.dropna(subset=['RainTomorrow', 'RainToday']) # drop row that doesn`t have target variable
    data["Date"] = pd.to_datetime(data['Date'])
    data['Year'] = data["Date"].dt.year
    data['Month'] = data["Date"].dt.month # group the data by Date(year,month)
    return data

# load the dataset that includes latitude and longitude of locations
# it needs when we calculate euclidean distance from location to location
def load_map_data():
    data_map = pd.read_csv('map.csv')
    data_map['latitude'] = pd.to_numeric(data_map['latitude'])
    data_map['longitude'] = pd.to_numeric(data_map['longitude'])
    return data_map

# function that fills missing values with the mean of their respective groups
def fill_nan_with_mean(data, grouped_data):
    for name, group in grouped_data:
        mean_values = group[features].mean() # input mean to missing values
        for feature in features:
            if group[feature].isna().all(): # Check if all the feature values of a group are missing value
                try:
                    # fill in with the mean of the same location-year-month case from the previous year
                    previous_year = grouped_data.get_group((name[0], name[1]-1, name[2]))[feature].mean()
                except KeyError:
                    # fill in with the mean of the same location-year-month case from the next year(exception)
                    previous_year = grouped_data.get_group((name[0], name[1]+1, name[2]))[feature].mean()
                data.loc[group.index, feature] = previous_year
            else:
                data.loc[group.index, feature] = data.loc[group.index, feature].fillna(mean_values[feature])
    return data

# Second preprocessing for features that still have missing values after the first preprocessing of 'features'.
# that are 'Pressure9am', 'Pressure3pm', 'WindGustSpeed' features.
def fill_nan_with_nearest_location(data, grouped_data, data_map):
    for name, group in grouped_data:
        for feature in features_2nd:
            if group[feature].isna().all(): # Check if all the feature values of a group are missing value
                # Get the latitude and longitude of the current location
                coords_1 = data_map[data_map['Location'] == name[0]][['latitude', 'longitude']].values[0]
                min_dist = float('inf')  # Initialize minimum distance to infinity
                nearest_loc = '' 
                # Loop through all the locations
                for i, row in data_map.iterrows():
                    if row['Location'] != name[0]:
                        coords_2 = [row['latitude'], row['longitude']] # Get the latitude and longitude of the location being compared
                        dist = distance.euclidean(coords_1, coords_2)  # Calculate Euclidean distance
                        if dist < min_dist: # Update minimum distance(nearest location)
                            min_dist = dist
                            nearest_loc = row['Location']
                try: # Try to get the mean value of the feature for the nearest location
                    nearest_year_month_val = grouped_data.get_group((nearest_loc, name[1], name[2]))[feature].mean()
                except KeyError:
                     # If the data for the nearest location is not available, fill with NaN
                    nearest_year_month_val = np.nan
                # Fill the missing values with the mean value of the nearest location   
                data.loc[group.index, feature] = nearest_year_month_val
    return data

# Function for handling exceptions for features that still have missing values even after being filtered through 'features_2nd'
# that are -> special_cases = [("Newcastle", 2008, [12]), ("Penrith", 2008, [12]), 
                            # ("Sydney", 2008, list(range(2, 13))), ("Albany", 2008, [12]), 
                            # ("Albany", 2012, [8, 9])]
def handle_special_cases(data, grouped_data):
    for loc, year, months in special_cases:
        for month in months:
            for feature in features_2nd: 
                try:
                    special_group = grouped_data.get_group((loc, year, month))
                    if special_group[feature].isna().all(): # Check if all the feature values of a group are missing value
                        try: # fill in with the mean of the same location-year-month case from the next year
                            next_year_month_val = grouped_data.get_group((loc, year+1, month))[feature].mean()
                            # input mean of the next year
                            data.loc[special_group.index, feature] = next_year_month_val
                        except KeyError:
                            # If the data for the next year is not available, skip to the next feature.
                            continue
                except KeyError:
                     # If the group for the specific location, year and month is not found, skip to the next group.
                    continue
    return data

# Function to fill the missing values of categorical features with the mode (most frequent value) of each group.
# These features have directional characteristics, so the mode was used instead of the mean to fill the missing values.
def fill_categorical_nan_with_mode(data, grouped_data):
    for name, group in grouped_data:
        for feature in categorical_features:
            if group[feature].isna().any(): # Check if all the feature values of a group are missing value
                mode_values = group[feature].mode()
                if len(mode_values) > 0: # At least there is one content in a group(not all NaN)
                    mode_value = mode_values[0] # Fill missing values with the mode value
                    data.loc[group.index, feature] = data.loc[group.index, feature].fillna(mode_value)
    return data

# Second preprocessing for features that still have missing values after the first preprocessing of 'categorical_features'.
def fill_categorical_nan_with_nearest_location(data, grouped_data, data_map):
    for name, group in grouped_data:
        for feature in categorical_features:
            if group[feature].isna().all(): # Check if all the feature values of a group are missing value
                # Find the nearest location
                coords_1 = data_map[data_map['Location'] == name[0]][['latitude', 'longitude']].values[0]
                # Initialize minimum distance to infinity
                min_dist = float('inf')
                nearest_loc = ''
                for i, row in data_map.iterrows(): # loop that iterates over the rows of the data_map DataFrame.
                    if row['Location'] != name[0]:
                        # assigning the values of the 'latitude' and 'longitude' columns
                        coords_2 = [row['latitude'], row['longitude']]
                        dist = distance.euclidean(coords_1, coords_2)
                        if dist < min_dist: # Update minimum distance(nearest location)
                            min_dist = dist
                            nearest_loc = row['Location']
                try:
                    # Try to fill the missing values with the mode value of the nearest location
                    nearest_year_month_val = grouped_data.get_group((nearest_loc, name[1], name[2]))[feature].mode()[0]
                except KeyError:
                    # If the data for the nearest location is not available, fill with NaN
                    nearest_year_month_val = np.nan
                # Fill the missing values with the mode value of the nearest location   
                data.loc[group.index, feature] = nearest_year_month_val
    return data

# Function for handling exceptions for features fill_categorical_nan_with_nearest_location function
# that still have missing values even after being filtered from 
# exception_cases = [('Sydney', 2008, range(1, 13)), 
                 #   ('Newcastle', 2008, [12]), 
                 #   ('Albany', 2008, [12]), 
                 #   ('Albany', 2012, [8, 9])]
def handle_exception_cases(data, grouped_data):
    for case in exception_cases:
        location, year, months = case[0], case[1], case[2]
        for month in months:
            for feature in categorical_features:
                # Check if all the feature values of a group are missing value
                if data[(data['Location'] == location) & (data['Year'] == year) & (data['Month'] == month)][feature].isna().all():
                    try:
                        # Try to fill with the mode value of the next year(same location,and month)
                        next_year_month_val = grouped_data.get_group((location, year+1, month))[feature].mode()[0]
                        data.loc[(data['Location'] == location) & (data['Year'] == year) & (data['Month'] == month), feature] = next_year_month_val
                    except KeyError:
                        try:
                            # If data for the next year is not available, try to fill with the mode value of the previous year
                            previous_year_month_val = grouped_data.get_group((location, year-1, month))[feature].mode()[0]
                            data.loc[(data['Location'] == location) & (data['Year'] == year) & (data['Month'] == month), feature] = previous_year_month_val
                        except KeyError:
                            # If data for the previous year is also not available, fill with NaN
                            data.loc[(data['Location'] == location) & (data['Year'] == year) & (data['Month'] == month), feature] = np.nan
    return data

# ordinalEncoding to Categorical data
def convert_to_categorical_and_encode(data):
    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
    enc = OrdinalEncoder()
    data[categorical_features] = enc.fit_transform(data[categorical_features])
    return data

# minmax scaling to data with no missing value
def scale_features(data):
    num_features = data.select_dtypes(include=[np.number]).drop(columns=["Year", "Month"]).columns # Select numeric features excluding Year and Month
    scaler = MinMaxScaler()  # Create a scaler
    data[num_features] = scaler.fit_transform(data[num_features]) # Apply scaling
    return data

def main():
    data = load_and_preprocess_data()
    data_map = load_map_data()
    grouped_data = data.groupby(['Location', 'Year', 'Month'])
    data = fill_nan_with_mean(data, grouped_data)
    data = fill_nan_with_nearest_location(data, grouped_data, data_map)
    data = handle_special_cases(data, grouped_data)
    data = fill_categorical_nan_with_mode(data, grouped_data)
    data = fill_categorical_nan_with_nearest_location(data, grouped_data, data_map)
    data = handle_exception_cases(data, grouped_data)
    data = convert_to_categorical_and_encode(data)
    data = scale_features(data)
    # Encoding 'rainTomorrow' and 'rainToday' [No: 0 , Yes: 1]
    data['RainTomorrow'] = data['RainTomorrow'].map({'No': 0, 'Yes': 1})
    data['RainToday'] = data['RainToday'].map({'No': 0, 'Yes': 1})
    data.to_csv('preprocessed_data.csv', index = False)
    print(data.isnull().sum()) # tells no missing values in data
    return data

if __name__ == "__main__":
    data = main()
    print(data)


