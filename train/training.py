import models.unet_model as unet

# create test data
import torch
import torch.nn as nn

def create_test_data(batch_size=1, channels=3, height=64, width=64):
    return torch.randn(batch_size, channels, height, width)

# put data on GPU if available
def create_test_data_gpu(batch_size=1, channels=3, height=64, width=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return create_test_data(batch_size, channels, height, width).to(device)

# function train the model
def train_model(model, data, epochs=1, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Assuming a classification task

    for epoch in range(epochs):
        # optimizer.zero_grad() resets the gradients of the model parameters
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)  # Adjust this according to your task
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# run the training

# Create a U-Net model
model = unet.UNet(in_channels=3, out_channels=1)  # Example with 3 input channels and 1 output channel

# Create test data
test_data = create_test_data_gpu(batch_size=4, channels=3, height=64, width=64)

# Train the model
train_model(model, test_data, epochs=5, learning_rate=0.001)