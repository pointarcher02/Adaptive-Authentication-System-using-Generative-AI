

import sys
import os

sys.path.append(r"c:\Users\insid\Downloads\Biovision slides\Thales\Thales")


import torch
import torch.optim as optim
from app.model import VAE
import data_prep

# Load and preprocess data
data = data_prep.preprocess_data()

# Set input dimensions to match the feature size of the data
input_dim = 3  # Features: typing speed, mouse movement, location
hidden_dim = 32
latent_dim = 8

# Instantiate the VAE
vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training loop
epochs = 100  # Increase epochs for better training
data_tensor = torch.tensor(data).float()

for epoch in range(epochs):
    optimizer.zero_grad()
    reconstructed = vae(data_tensor)
    loss = ((data_tensor - reconstructed) ** 2).mean()  # Reconstruction loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the model
torch.save(vae, "scripts/model.pth")

