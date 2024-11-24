# app/model.py
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x

    @staticmethod
    def load_model(filepath):
        # Load model from file
        model = torch.load(filepath)
        model.eval()
        return model


def check_anomaly(data, model):
    # Extract feature values from the input dictionary
    location_encoded = hash(data["location"]) % 100  # Simplified encoding for location, adjust as needed

    input_values = [data["typing_speed"], data["mouse_movement"], location_encoded]

    # Convert to tensor
    input_data = torch.tensor(input_values).float()

    # Pass through the VAE model
    output = model(input_data)

    # Calculate reconstruction error
    reconstruction_error = ((input_data - output) ** 2).mean().item()

    return reconstruction_error


