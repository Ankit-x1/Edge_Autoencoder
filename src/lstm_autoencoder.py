import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, latent_dim=16, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim,  
            num_layers=num_layers, 
            batch_first=True
        )
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1]  
        z = self.latent(h).unsqueeze(1)  

        decoded_input = self.decoder_input(z).repeat(1, x.size(1), 1)
        decoded_output, _ = self.decoder(decoded_input)
        reconstructed = self.decoder_output(decoded_output)

        return reconstructed
