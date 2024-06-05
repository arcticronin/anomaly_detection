import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchsummary
import numpy as np

from torch.utils.data import Dataset, DataLoader

from kneed import KneeLocator

# custom
import importlib
import utils
importlib.reload(utils) #debug - remove
import preprocessing
importlib.reload(preprocessing) #debug - remove
import models
importlib.reload(models) #debug - remove

class DataFrameDataset(Dataset):
    """
    Custom Dataset for loading a pandas DataFrame, used for a PyTorch DataLoader
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data
    """
    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a sample from the dataset
        return self.data[idx]


def create_dataloader(df, batch_size=1, shuffle=True):
    """
    Create a DataLoader using the DataFrameDataset class
    Args:
        df (pd.DataFrame): DataFrame containing the data
        batch_size (int): Number of samples per batch
        shuffle (bool): boolean to shuffle the data
    """
    dataset = DataFrameDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def main(dataframe):
    torch.manual_seed(99)
    df = dataframe
    data_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
    binary_indices = utils.binary_indices
    continuous_indices = utils.continuous_indices

    # setting batch size of 32, and shuffle to True
    # the training has been done on CPU
    dataloader = create_dataloader(df, batch_size=32, shuffle=True)

    # create the model using the Autoencoder with latent space = 3
    model = models.Autoencoder_Encoder(binary_indices=binary_indices)

    # setting training parameters
    epochs = 50
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = models.Autoencoder_Loss_Prob(binary_indices=binary_indices,
                                             continuous_indices=continuous_indices)
    # training loop
    for epoch in range(epochs):
        for data in dataloader:
            model.train()
            optimizer.zero_grad()
            # get reconstructed data
            x_reconstructed = model(data)
            # compute the loss against the original data
            loss = criterion(data, x_reconstructed)
            # backpropagation
            loss.backward()
            optimizer.step()
        # Step the scheduler
        scheduler.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]}')
    print("Training complete")
    # sort reconstruction errors
    distances = [criterion(data_tensor[i, :].unsqueeze(0),
                           model(data_tensor)[i, :].unsqueeze(0)).item() for i in range(len(df))]

    # find the knee point
    sorted_distances = np.sort(distances)
    knee = KneeLocator(range(len(sorted_distances)),
                       sorted_distances,
                       curve='convex',
                       direction='increasing',
                       S= 2.3)
    treshold = knee.knee_y
    #print(f"treshold is: {treshold:.4f}")
    # flag outliers as -1, inliers as 0
    outlier_index = [-1 if i > treshold / 2 else 0 for i in distances]
    #print(f"percentage of outliers is: {-np.sum(outlier_index) / len(outlier_index) * 100: .2f}%")
    print(f"number of outliers is: {-np.sum(outlier_index)}")

    return outlier_index


if __name__ == '__main__':
    main()
