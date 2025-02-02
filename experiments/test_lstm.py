import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from algorithms.lstm.dataset import LSTMDataLoader
from algorithms.lstm.network import LSTMModel
from algorithms.lstm.utils import calculate_next_state
import matplotlib.gridspec as gridspec
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
    hidden_size = 512
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

    # Load the trained model
    model_path = 'experiments/models/LSTM_LR_0.001_HSZ_512_NL_2_2025-02-02_01-56-10.pth'  # Replace with your model path
    model = load_model(model_path, input_size, hidden_size, output_size, num_layers, device)

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
    num_steps = 1000
    # Perform rolling predictions
    predictions = rolling_predictions(model, initial_window, test_data, num_steps, device)
    
    
    # Extract ground truth for comparison
    ground_truth = []
    for _, Y_test in test_loader:
        ground_truth.append(Y_test.cpu().numpy())
    ground_truth = np.concatenate(ground_truth, axis=0)[:num_steps]  # Shape: [num_steps, output_size]

    
    # Calculate evaluation metrics
    filtered_predictions = np.delete(predictions, [0,1,2,3,4,5,8,12,13,14,15], axis=1)  # Remove time columns
    
    mae = mean_absolute_error(ground_truth, filtered_predictions)
    mse = mean_squared_error(ground_truth, filtered_predictions)
    rmse = np.sqrt(mse)
    
    predictions_original = filtered_predictions * stds['y'] + means['y']
    ground_truth_original = ground_truth * stds['y'] + means['y']
    
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    # Create grid of subplots for 5 predictions (adjust based on your actual number of predictions)
    num_predictions = filtered_predictions.shape[1]  # Number of predictions (variables)

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    spec = gridspec.GridSpec(nrows=2, ncols=6, figure=fig)  # 6 columns for even spacing

    # Define subplot positions
    ax1 = fig.add_subplot(spec[0, 0:2])  # First row, first 2 columns
    ax2 = fig.add_subplot(spec[0, 2:4])  # First row, middle 2 columns
    ax3 = fig.add_subplot(spec[0, 4:6])  # First row, last 2 columns
    ax4 = fig.add_subplot(spec[1, 1:3])  # Second row, center-left (spanning 2 columns)
    ax5 = fig.add_subplot(spec[1, 3:5])  # Second row, center-right (spanning 2 columns)

    # Store axes for iteration
    axes = [ax1, ax2, ax3, ax4, ax5]

    labels = ['Inside Temperature', 'Inside Humidity', 'Inside CO2 Concentration', 'Window Fan Energy Consumption', 'HVAC Energy Consumption']
    labels_y = ['Temperature (°C)', 'Humidity (%)', 'CO2 (ppm)', 'Energy Consumption (kWh)', 'Energy Consumption (kWh)']

    # Plot each variable in its corresponding subplot
    for i in range(num_predictions):
        axes[i].plot(ground_truth_original[:, i], label='Ground Truth', linestyle='-', color='blue',alpha=1.0)
        axes[i].plot(predictions_original[:, i], label='Predictions', linestyle='dashed', color='red',alpha=0.8)  # Dashed line for predictions
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel(labels_y[i])
        axes[i].set_title(labels[i])
        # Move legend outside the plot (right side)
    fig.legend(labels=['Ground Truth', 'Predictions'], loc="lower center", ncol=2)    
    plt.tight_layout()  # Leaves space for legends
    # Adjust layout for better spacing
    

    # Save and show
    plt.savefig('rolling_test_grid.png')
    plt.show()
    plt.close()