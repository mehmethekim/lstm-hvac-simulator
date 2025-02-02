import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class LSTMDataLoader:
    def __init__(self, csv_files, window_size=10, batch_size=64, validation_size=0.15, test_size=0.15):
        self.csv_files = csv_files
        self.window_size = window_size
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.test_size = test_size
    def create_sequences(self, df):
        """
        Create input-output sequences for the LSTM.
        - Input (X): Includes all columns except 'timestamp'.
        - Output (Y): Excludes 'month', 'day_of_month', and 'hour'.
        """
        # Convert DataFrame to NumPy array for faster operations
        data = df.values  # Shape: [num_samples, num_features]
        # column_names_X = ['month', 'day_of_month', 'hour', 'minute', 'outdoor_temperature', 
        #               'outdoor_humidity', 'air_temperature', 'air_humidity', 
        #               'people_occupant', 'air_co2', 'window_fan_energy', 
        #               'total_electricity_HVAC', 'heating_setpoint', 
        #               'cooling_setpoint', 'ac_fan_speed', 'window_fan_speed']
        # Columns to exclude from the output (Y)
        exclude_columns = ['month', 'day_of_month', 'hour','minute','outdoor_temperature',
                        'outdoor_humidity','people_occupant','heating_setpoint', 
                        'cooling_setpoint', 'ac_fan_speed', 'window_fan_speed']
        exclude_indices = [df.columns.get_loc(col) for col in exclude_columns]  # Get indices of columns to exclude
        
        # Prepare input (X) and output (Y) arrays
        num_samples = len(data) - self.window_size
        num_features_X = data.shape[1]  # All features
        num_features_Y = data.shape[1] - len(exclude_indices)  # Exclude 'month', 'day_of_month', 'hour'
        
        X = np.zeros((num_samples, self.window_size, num_features_X))  # Shape: [num_samples, window_size, num_features_X]
        Y = np.zeros((num_samples, num_features_Y))  # Shape: [num_samples, num_features_Y]
        
        # Fill X and Y using vectorized operations
        for i in range(num_samples):
            X[i] = data[i:i + self.window_size]  # Input window
            Y[i] = np.delete(data[i + self.window_size], exclude_indices)  # Output, excluding specified columns
        
        return X, Y
    def load_data(self):
        train_val_sequences = []  # List to store all input-output pairs
        test_sequences = []
        
        for csv_file in self.csv_files:
            df = pd.read_csv(csv_file)
            
            # Create sequences for each file
            X, Y = self.create_sequences(df)
    
            train_val_sequences.append((X[:-int(self.test_size * len(X))], Y[:-int(self.test_size * len(Y))]))
            test_sequences.append((X[-int(self.test_size * len(X)):], Y[-int(self.test_size * len(Y)):]))
        
        
        # Combine sequences from all files
        X_train_and_val = np.concatenate([x for x, _ in train_val_sequences], axis=0)
        Y_train_and_val = np.concatenate([y for _, y in train_val_sequences], axis=0)
        
        X_test = np.concatenate([x for x, _ in test_sequences], axis=0)
        Y_test = np.concatenate([y for _, y in test_sequences], axis=0)
        
        x_mean = np.mean(X_train_and_val, axis=(0, 1))  # mean across window and features
        x_std = np.std(X_train_and_val, axis=(0, 1))    # std across window and features
        
        y_mean = np.mean(Y_train_and_val, axis=0)  # mean across features
        y_std = np.std(Y_train_and_val, axis=0)    # std across features
        
        x_std = np.where(x_std == 0, 1e-6, x_std) 
        y_std = np.where(y_std == 0, 1e-6, y_std) 
        # Normalize data
        X_train_and_val = (X_train_and_val - x_mean) / x_std
        Y_train_and_val = (Y_train_and_val - y_mean) / y_std
        
        X_test = (X_test - x_mean) / x_std
        Y_test = (Y_test - y_mean) / y_std
        
        means = {'x': x_mean, 'y': y_mean}
        stds = {'x': x_std, 'y': y_std}
        # Split into train, validation, and test sets
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_and_val, Y_train_and_val, test_size=self.validation_size, shuffle=True)
        
        
        # Create DataLoader
        train_loader = DataLoader(list(zip(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))),
                                  batch_size=self.batch_size, shuffle=True)
        
        val_loader = DataLoader(list(zip(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))),
                                batch_size=self.batch_size, shuffle=False)
        
        test_loader = DataLoader(list(zip(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))),
                                 batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, means, stds

class LSTMDataset(Dataset):
    def __init__(self, filepaths, window_size=10):
        self.window_size = window_size
        self.data = self.load_data(filepaths)

    def load_data(self, filepaths):
        sequences = []  # List to store all input-output pairs
        
        for filepath in filepaths:
            df = pd.read_csv(filepath)
            
            # Remove timestamp column and irrelevant columns like heating/cooling settings, fan speeds, etc.
            df = df.drop(columns=['timestamp'])
            
            # Create sequences for each file
            X, Y = self.create_sequences(df)
            
            sequences.append((X, Y))
        
        # Combine sequences from all files
        X_all = np.concatenate([x for x, _ in sequences], axis=0)
        Y_all = np.concatenate([y for _, y in sequences], axis=0)
        
        return X_all, Y_all

    def create_sequences(self, df):
        X = []
        Y = []
        
        # Iterate over the data with a sliding window approach
        for i in range(len(df) - self.window_size):
            window = df.iloc[i:i + self.window_size]  # Create the window of 10 observations
            
            # Input should not include 'timestamp'
            input_data = window.values
            
            # Output should be the next observation (next row)
            next_observation = df.iloc[i + self.window_size].values
            
            X.append(input_data)
            Y.append(next_observation)

        return np.array(X), np.array(Y)

    def __len__(self):
        # Number of sequences in the dataset
        return len(self.data[0])
    
    def __getitem__(self, idx):
        X, Y = self.data
        return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(Y[idx], dtype=torch.float32)

# # Usage example:
# filepaths = ['simulation_data_1.csv', 'simulation_data_2.csv', 'simulation_data_3.csv', 'simulation_data_4.csv']

# # Create the LSTM dataset
# dataloader = LSTMDataLoader(filepaths, window_size=10, batch_size=64)

# # Create DataLoader to feed batches of data to the LSTM
# train_loader, val_loader, test_loader, mean, std = dataloader.load_data()
# # Print mean and std
# print("Mean:", mean)
# print("Standard Deviation:", std)

# # Example: Accessing a batch from the train_loader
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(f"Batch {batch_idx}:")
#     print("Data:", data[0])
#     print("Target:", target[0])
#     break  # For testing, break after first batch

