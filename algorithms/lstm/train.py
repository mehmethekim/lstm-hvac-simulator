from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from datetime import datetime

def train_lstm(model, train_loader, val_loader, criterion, optimizer, device):
    train_losses = []
    val_losses = []
    
    
    model.train()
    running_train_loss = 0.0
    
    for batch in train_loader:
        # Get data
        X_train, Y_train = batch
        X_train, Y_train = X_train.to(device), Y_train.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        
        # Compute loss
        loss = criterion(outputs, Y_train)
            
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation loss
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            X_val, Y_val = batch
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            
            # Forward pass
            outputs = model(X_val)
            
            # Compute loss
            loss = criterion(outputs, Y_val)
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    

    return avg_train_loss, avg_val_loss