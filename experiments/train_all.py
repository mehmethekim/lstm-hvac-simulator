from algorithms.lstm.dataset import *
from algorithms.lstm.train import *
from algorithms.lstm.network import *
import os
# Ensure directories exist
models_dir = "experiments/models"
plots_dir = "experiments/plots"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

filepaths = ['simulation_data_1.csv', 'simulation_data_2.csv', 'simulation_data_3.csv', 'simulation_data_4.csv']

# Create the LSTM dataset
dataloader = LSTMDataLoader(filepaths, window_size=10, batch_size=64)

# Create DataLoader to feed batches of data to the LSTM
train_loader, val_loader, test_loader, means, stds = dataloader.load_data()

# Get the shape of the first batch from train_loader to determine input_size
for X_train, Y_train in train_loader:
    # Assume that the first batch is enough to get input_size and output_size
    input_size = X_train.shape[2]  # (batch_size, sequence_length, input_size)
    output_size = Y_train.shape[1]  # (batch_size, output_size)
    break  # We only need the first batch to get the shape
# -------------------------
# Hyperparameter Grid
# -------------------------
learning_rates = [0.001, 0.0005, 0.0001]
hidden_sizes = [1024]
num_layers_list = [1, 2, 3]

epochs = 100
patience = 5  # Early stopping patience

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Grid Search Training Loop
# -------------------------
for learning_rate in learning_rates:
    for hidden_size in hidden_sizes:
        for num_layers in num_layers_list:

            print(f"\nTraining configuration: lr={learning_rate}, hidden_size={hidden_size}, num_layers={num_layers}")
            # Initialize model, criterion, and optimizer
            model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
            criterion = nn.MSELoss()  # For regression tasks, use MSELoss
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            # Train the model
            # Early Stopping setup
            best_val_loss = float('inf')
            patience = 5  # Number of epochs to wait before stopping
            epochs_without_improvement = 0
            train_losses = []
            val_losses = []
            for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
                avg_train_loss, avg_val_loss = train_lstm(model, train_loader, val_loader, criterion, optimizer, device)
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                tqdm.write(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print("Early stopping")
                    break

                scheduler.step()
            # Save the trained model and plot for this configuration
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_filename = os.path.join(models_dir, f"LSTM_LR_{learning_rate}_HSZ_{hidden_size}_NL_{num_layers}_{current_time}.pth")
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved to {model_filename}")
            
            # Plot training vs. validation loss
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f"Train vs Validation Loss\nlr={learning_rate}, hidden_size={hidden_size}, num_layers={num_layers}")
            plt.legend()
            plot_filename = os.path.join(plots_dir, f"loss_plot_LSTM_LR_{learning_rate}_HSZ_{hidden_size}_NL_{num_layers}_{current_time}.png")
            plt.savefig(plot_filename)
            plt.close()