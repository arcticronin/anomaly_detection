from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):

    def __init__(self, binary_indices, input_dim=21):
        """
        Autoencoder module.
        Args:
            binary_indices (list of int): Indices of binary features.
            continuous_indices (list of int): Indices of continuous features.
            input_dim (int): Number of input features.
        """
        super(Autoencoder, self).__init__()
        self.binary_indices = binary_indices

        # Encoder with latent space = 4
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16),
            nn.ReLU(),
            nn.Dropout(0.1),
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
            nn.Dropout(0.1),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        # encodes and decodes input
        x_encoded = self.encoder(x)
        x_reconstructed = self.decoder(x_encoded)
        return x_reconstructed

    ## TODO remove if not used
    # def post_process(self, x_reconstructed):
    #     # applies sigmoid activation to binary features only
    #     # we preferred to use a sigmoid function , but can be an option when the model is trained
    #     x_reconstructed[:, self.binary_indices] = torch.sigmoid(x_reconstructed[:, self.binary_indices])
    #     return x_reconstructed


class Autoencoder_Loss_Prob(nn.Module):
    """
    Custom loss function for autoencoder training with binary and continuous data
    """
    def __init__(self, binary_indices, continuous_indices):
        """
        Initializes the Autoencoder_Loss_Prob module.
        Args:
            binary_indices (list of int): Indices of binary features.
            continuous_indices (list of int): Indices of continuous features.
        """
        super(Autoencoder_Loss_Prob, self).__init__()
        self.binary_indices = binary_indices
        self.continuous_indices = continuous_indices
        self.alpha = len(binary_indices) / (len(binary_indices) + len(continuous_indices))

    def forward(self, x_original, x_reconstructed):
        """
        Computes the weighted loss for autoencoder reconstruction.

        Args:
            x_original (torch.Tensor): Original input tensor.
            x_reconstructed (torch.Tensor): Reconstructed input tensor.

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Extract binary features
        x_bin_original = x_original[:, self.binary_indices]
        x_bin_reconstructed = x_reconstructed[:, self.binary_indices]

        # Compute binary cross-entropy loss for binary features
        binary_loss = F.binary_cross_entropy_with_logits(
            x_bin_reconstructed, x_bin_original, reduction='mean'
        )

        # Extract continuous features
        x_cont_original = x_original[:, self.continuous_indices]
        x_cont_reconstructed = x_reconstructed[:, self.continuous_indices]

        # Compute mean squared error loss for continuous features
        continuous_loss = F.mse_loss(x_cont_reconstructed, x_cont_original, reduction='mean')

        # Combine losses with respective weights
        total_loss = self.alpha * binary_loss + (1 - self.alpha) * continuous_loss
        return total_loss

## Autoencoder with 2D and 3D latent spaces, used for visualization

class Autoencoder_Encoder(nn.Module):
    def __init__(self, binary_indices, input_dim=21):
        super(Autoencoder_Encoder, self).__init__()
        self.binary_indices = binary_indices

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        return self.decoder(x_encoded)
    def encode(self, x):
        return self.encoder(x)


class Autoencoder_2D_Encoder(nn.Module):
    def __init__(self, binary_indices, input_dim=21):
        super(Autoencoder_2D_Encoder, self).__init__()
        self.binary_indices = binary_indices

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        return self.decoder(x_encoded)
    def encode(self, x):
        return self.encoder(x)


