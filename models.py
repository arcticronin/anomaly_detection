from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder_OLD(nn.Module):
    def __init__(self, binary_indices):
        super(Autoencoder_OLD, self).__init__()
        self.binary_indices = binary_indices

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(21, 16),
            nn.ReLU(),
            nn.Linear( 16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 21)
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_reconstructed = self.decoder(x_encoded)
        return x_reconstructed

class GowerLoss_OLD(nn.Module):
    def __init__(self, binary_indices, continuous_indices):
        super(GowerLoss_OLD, self).__init__()
        self.binary_indices = binary_indices
        self.continuous_indices = continuous_indices
        self.alpha = len(binary_indices) / (len(binary_indices) + len(continuous_indices))

    def forward(self, x_original, x_reconstructed):
        # Binary indices handling
        x_bin_original = (x_original[:, self.binary_indices] >= 0.5).int()
        x_bin_reconstructed = (x_reconstructed[:, self.binary_indices] >= 0.5).int()

        # Binary loss (counting differing elements)
        #binary_loss = (torch.sum(x_bin_reconstructed != x_bin_original) / len(self.binary_indices)).float()
        binary_loss = (torch.abs(x_bin_reconstructed - x_bin_original)).float().mean()

        # Continuous data handling
        x_cont_original = x_original[:, self.continuous_indices]
        x_cont_reconstructed = x_reconstructed[:, self.continuous_indices]

        # Continuous loss (using Mean Absolute Error)
        continuous_loss = F.l1_loss(x_cont_reconstructed, x_cont_original, reduction='mean')

        # Combine losses
        total_loss = self.alpha * binary_loss + (1 - self.alpha) * continuous_loss
        return total_loss

class PCA_Autoencoder(nn.Module):
    def __init__(self, input_dim, binary_indices):
        super(PCA_Autoencoder, self).__init__()
        self.binary_indices = binary_indices
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16),
            nn.Linear(16, 8),
            nn.Linear(8, 4)  # Assuming 4 principal components
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.Linear(8, 16),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class PCA_Encoder(nn.Module):
    def __init__(self, input_dim, binary_indices):
        super(PCA_Encoder, self).__init__()
        self.binary_indices = binary_indices
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16),
            nn.Linear(16, 8),
            nn.Linear(8, 4),
            nn.linear(4,2)# Assuming 4 principal components
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded



class Autoencoder(nn.Module):
    def __init__(self, binary_indices, input_dim=21):
        super(Autoencoder, self).__init__()
        self.binary_indices = binary_indices

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_reconstructed = self.decoder(x_encoded)
        return self.post_process(x_reconstructed)

    def post_process(self, x_reconstructed):
        # Apply sigmoid activation to binary features only
        x_reconstructed[:, self.binary_indices] = torch.sigmoid(x_reconstructed[:, self.binary_indices])
        return x_reconstructed


class GowerLoss_Prob(nn.Module):
    def __init__(self, binary_indices, continuous_indices, pos_weight):
        super(GowerLoss_Prob, self).__init__()
        self.binary_indices = binary_indices
        self.continuous_indices = continuous_indices
        self.alpha = len(binary_indices) / (len(binary_indices) + len(continuous_indices))
        self.pos_weight = pos_weight

    def forward(self, x_original, x_reconstructed):
        # Binary indices handling
        x_bin_original = x_original[:, self.binary_indices]
        x_bin_reconstructed = x_reconstructed[:, self.binary_indices]

        # Weighted binary cross-entropy loss
        binary_loss = F.binary_cross_entropy(
            x_bin_reconstructed, x_bin_original, pos_weight=self.pos_weight, reduction='mean'
        )

        # Continuous data handling
        x_cont_original = x_original[:, self.continuous_indices]
        x_cont_reconstructed = x_reconstructed[:, self.continuous_indices]

        # Continuous loss (using Mean Absolute Error)
        continuous_loss = F.mse_loss(x_cont_reconstructed, x_cont_original, reduction='mean')

        # Combine losses
        total_loss = self.alpha * binary_loss + (1 - self.alpha) * continuous_loss
        return total_loss