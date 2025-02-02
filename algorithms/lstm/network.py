import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.init as init
from tqdm import tqdm

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                init.xavier_uniform_(param.data)  # Xavier Initialization
            elif 'weight_hh' in name:  # Hidden-hidden weights
                init.orthogonal_(param.data)  # Orthogonal Initialization
            elif 'bias' in name:  # Biases
                init.zeros_(param.data)  # Initialize biases to zeros

        # Initialize fully connected layer weights
        init.xavier_uniform_(self.fc.weight.data)  # Xavier initialization for FC weights
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias.data)  # Initialize FC bias to zeros
    
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the output from the last timestep for prediction
        out = self.fc(out[:, -1, :])
        return out