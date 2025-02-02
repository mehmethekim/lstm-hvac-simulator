import torch
def test_lstm(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():  # No need to calculate gradients during testing
        for inputs, targets in test_loader:
            # Move data to the specified device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    # Calculate average test loss
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss