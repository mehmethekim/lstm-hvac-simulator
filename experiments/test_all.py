import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from algorithms.lstm.dataset import LSTMDataLoader
from algorithms.lstm.network import LSTMModel
from algorithms.lstm.utils import calculate_next_state
import os
import pandas as pd
column_names_X = ['month', 'day_of_month', 'hour', 'minute', 'outdoor_temperature', 
                  'outdoor_humidity', 'air_temperature', 'air_humidity', 
                  'people_occupant', 'air_co2', 'window_fan_energy', 
                  'total_electricity_HVAC', 'heating_setpoint', 
                  'cooling_setpoint', 'ac_fan_speed', 'window_fan_speed']

exclude_columns = ['month', 'day_of_month', 'hour', 'minute', 'outdoor_temperature',
                   'outdoor_humidity', 'people_occupant', 'heating_setpoint', 
                   'cooling_setpoint', 'ac_fan_speed', 'window_fan_speed']

# Define columns to predict (Y) by removing excluded columns
column_names_Y = [col for col in column_names_X if col not in exclude_columns]

# Get indices of columns to predict (Y) in test dataset
column_indices_Y = [column_names_X.index(col) for col in column_names_Y]
# Extract indices of predicted variables
column_names_Y = [col for col in column_names_X if col not in exclude_columns]
# Load your trained model
def load_model(model_path, input_size, hidden_size, output_size, num_layers, device):
    """
    Load the trained LSTM model.
    """
    model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model


def rolling_predictions(model, initial_window, test_data, num_steps, device):
    """
    Perform rolling predictions while keeping external variables from the test dataset.

    Args:
        model: Trained LSTM model.
        initial_window: Initial window of data (shape: [sequence_length, input_size]).
        test_data: The actual test dataset (to provide external variables).
        num_steps: Number of steps to predict into the future.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        predictions: Array of predicted states with corrected external inputs.
    """
    predictions = []
    current_window = initial_window.clone().to(device)  # Shape: [sequence_length, input_size]
    
    for step in range(num_steps):
        # Extract last observation's timestamp
        last_observation = current_window[-1].cpu().numpy()
        month, day, hour, minute = last_observation[:4]  # Extract time values
        #print_test_data shape
        
        
        # Retrieve true external conditions from test_data
        if step < len(test_data):
            external_values = test_data[step, :].copy()  # Use test dataset
        else:
            external_values = predictions[-1]  # Use last prediction if out of range

        # Predict next state using LSTM
        with torch.no_grad():
            next_state = model(current_window.unsqueeze(0))  # Shape: [1, output_size]
        next_state = next_state.squeeze(0).cpu().numpy()

        # Replace only the predicted values in `external_values`
        external_values[column_indices_Y] = next_state  # Keep external values unchanged

        # Store updated row in predictions
        predictions.append(external_values)

        # Prepare input for next iteration
        next_state_tensor = torch.tensor(external_values, dtype=torch.float32, device=device)

        # Update rolling window
        current_window = torch.cat([current_window[1:], next_state_tensor.unsqueeze(0)], dim=0)

    return np.array(predictions)
# Main script
if __name__ == "__main__":
    # Parameters (should match training parameters)
    window_size = 10
    hidden_size = 256
    num_layers = 2
    input_size = None  # Will be determined from the dataset
    output_size = None  # Will be determined from the dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the test dataset
    filepaths = ['simulation_data_1.csv', 'simulation_data_2.csv', 'simulation_data_3.csv', 'simulation_data_4.csv']
    dataloader = LSTMDataLoader(filepaths, window_size=window_size, batch_size=64)
    train_loader, val_loader, test_loader, means, stds = dataloader.load_data()

    # Determine input_size and output_size from the test dataset
    for X_test, Y_test in test_loader:
        input_size = X_test.shape[2]  # (batch_size, sequence_length, input_size)
        output_size = Y_test.shape[1]  # (batch_size, output_size)
        break

    

    # Prepare initial window and collect the entire test dataset's external variables
    external_list = []
    initial_window = None
    for X_test, Y_test in test_loader:
        if initial_window is None:
            # Use the first sequence of the first batch as the initial window
            initial_window = X_test[0]  # Shape: [sequence_length, input_size]
        # For each batch, extract external variables from the last timestep of every sequence
        batch_external = X_test[:, -1, :].cpu().numpy()  # Shape: (batch_size, input_size)
        external_list.append(batch_external)
    # Concatenate all batches to form a continuous test dataset
    test_data = np.concatenate(external_list, axis=0)  # Shape: (total_timesteps, input_size)
    # Number of steps to predict into the future (e.g., 10000)
    num_steps = 300
    # Extract ground truth for comparison (concatenate from all test batches)
    ground_truth_list = []
    for _, Y_test in test_loader:
        ground_truth_list.append(Y_test.cpu().numpy())
    ground_truth = np.concatenate(ground_truth_list, axis=0)[:num_steps]  # use same num_steps as below

    # Directory where models are stored and CSV will be saved
    models_dir = "experiments/models"
    results_csv = "experiments/model_evaluation_results.csv"
        # Get list of model files in the models_dir
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    results = []  # List to store evaluation results
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        # Parse hyperparameters from filename; expected format:
        # "LSTM_LR_{lr}_HSZ_{hidden_size}_NL_{num_layers}_{timestamp}.pth"
        parts = model_file.split("_")
        try:
            learning_rate_val = float(parts[2])
            hidden_size_val = int(parts[4])
            num_layers_val = int(parts[6])
        except Exception as e:
            print(f"Skipping {model_file} due to parsing error: {e}")
            continue
        
        # Load the model using parsed hyperparameters
        model = load_model(model_path, input_size, hidden_size_val, output_size, num_layers_val, device)
        
        # Perform rolling predictions
        predictions = rolling_predictions(model, initial_window, test_data, num_steps, device)
        
        # Remove columns that are not predicted (as in your test script)
        filtered_predictions = np.delete(predictions, [0,1,2,3,4,5,8,12,13,14,15], axis=1)
        
        # Calculate evaluation metrics on normalized values
        mae = mean_absolute_error(ground_truth, filtered_predictions)
        mse = mean_squared_error(ground_truth, filtered_predictions)
        rmse = np.sqrt(mse)
        
        results.append({
            "model_file": model_file,
            "learning_rate": learning_rate_val,
            "hidden_size": hidden_size_val,
            "num_layers": num_layers_val,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        })
        
        print(f"Evaluated {model_file}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")
    #predictions = rolling_predictions(model, initial_window, test_data, num_steps, device)
    
    # Save the results to CSV
    df = pd.DataFrame(results)
    df.to_csv(results_csv, index=False)
    print(f"Evaluation results saved to {results_csv}")
    # # Extract ground truth for comparison
    # ground_truth = []
    # for _, Y_test in test_loader:
    #     ground_truth.append(Y_test.cpu().numpy())
    # ground_truth = np.concatenate(ground_truth, axis=0)[:num_steps]  # Shape: [num_steps, output_size]

    
    # # Calculate evaluation metrics
    # filtered_predictions = np.delete(predictions, [0,1,2,3,4,5,8,12,13,14,15], axis=1)  # Remove time columns
    
    # mae = mean_absolute_error(ground_truth, filtered_predictions)
    # mse = mean_squared_error(ground_truth, filtered_predictions)
    # rmse = np.sqrt(mse)
    
    # predictions_original = filtered_predictions * stds['y'] + means['y']
    # ground_truth_original = ground_truth * stds['y'] + means['y']
    
    # print(f'Mean Absolute Error (MAE): {mae:.4f}')
    # print(f'Mean Squared Error (MSE): {mse:.4f}')
    # print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    # # Create grid of subplots for 5 predictions (adjust based on your actual number of predictions)
    # num_predictions = filtered_predictions.shape[1]  # Number of predictions (variables)

    # # Create a grid with 2 rows and 3 columns to fit 5 plots (adjust grid size as necessary)
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # axes = axes.flatten()  # Flatten the axes array for easier indexing

    # labels = ['Air Temperature', 'Air Humidity', 'Air CO2', 'Window Fan Energy', 'Total Electricity HVAC']
    # # Plot each variable on its own subplot
    # for i in range(min(num_predictions, len(axes))):  # Avoid indexing errors if less than 5 variables
    #     axes[i].plot(ground_truth_original[:, i], label='Ground Truth')
    #     axes[i].plot(predictions_original[:, i], label='Predictions')
    #     axes[i].set_xlabel('Time Steps')
    #     axes[i].set_ylabel(f'Observation {i+1}')
    #     axes[i].set_title(labels[i])
    #     axes[i].legend()

    # # Remove any empty subplots (in case there are fewer than 5 variables)
    # for j in range(num_predictions, len(axes)):
    #     fig.delaxes(axes[j])

    # # Adjust layout to avoid overlapping subplots
    # plt.tight_layout()

    # # Save the plot
    # plt.savefig('rolling_test_grid.png')  # Save the grid plot as a PNG
    # plt.show()  # Show the plot
    # plt.close()  # Close plot to free memory