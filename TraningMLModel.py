import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Load the data we generated
positions = np.load('silica_positions.npy')
energies = np.load('silica_energies.npy')

# Flatten the position data for simplicity
# In a real application, you might use a more sophisticated representation
# like graph neural networks to preserve the 3D structure
n_samples = positions.shape[0]
positions_flat = positions.reshape(n_samples, -1)

# Create a PyTorch dataset
class MDDataset(Dataset):
    def __init__(self, positions, energies):
        self.positions = torch.tensor(positions, dtype=torch.float32)
        self.energies = torch.tensor(energies, dtype=torch.float32)
    
    def __len__(self):
        return len(self.energies)
    
    def __getitem__(self, idx):
        return self.positions[idx], self.energies[idx]

# Create data loaders
dataset = MDDataset(positions_flat, energies)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define a simple neural network model for energy prediction
class EnergyModel(nn.Module):
    def __init__(self, input_dim):
        super(EnergyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x).squeeze()

# Initialize model, loss function, and optimizer
input_dim = positions_flat.shape[1]
model = EnergyModel(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for positions, energies in train_loader:
        optimizer.zero_grad()
        outputs = model(positions)
        loss = criterion(outputs, energies)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for positions, energies in test_loader:
            outputs = model(positions)
            loss = criterion(outputs, energies)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Plot training results
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig('training_results.png')
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'silica_energy_model.pt')

# Make predictions on the test set
model.eval()
predictions = []
true_values = []

with torch.no_grad():
    for positions, energies in test_loader:
        outputs = model(positions)
        predictions.extend(outputs.numpy())
        true_values.extend(energies.numpy())

# Plot predictions vs true values
plt.figure(figsize=(10, 6))
plt.scatter(true_values, predictions, alpha=0.5)
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
plt.xlabel('True Energy (kJ/mol)')
plt.ylabel('Predicted Energy (kJ/mol)')
plt.title('Predicted vs True Energy Values')
plt.savefig('prediction_results.png')
plt.show()

print("Model training and evaluation complete!")