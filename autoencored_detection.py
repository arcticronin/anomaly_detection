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
    def __init__(self, dataframe):
        # Assuming dataframe is already scaled and prepared for neural network input
        self.data = torch.tensor(dataframe.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a sample from the dataset
        return self.data[idx]


def create_dataloader(df, batch_size=1, shuffle=True):
    dataset = DataFrameDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def main():
    ## set seed for reproducibility
    torch.manual_seed(99)
    df = preprocessing.load_dataset()
    data_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
    binary_indices = utils.binary_indices
    continuous_indices = utils.continuous_indices

    dataloader = create_dataloader(df, batch_size=32, shuffle=True)

    model = models.Autoencoder_Encoder(binary_indices=binary_indices)

    epochs = 50
    importlib.reload(utils)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = models.Autoencoder_Loss_Prob(binary_indices=binary_indices,
                                             continuous_indices=continuous_indices)

    for epoch in range(epochs):
        for data in dataloader:
            model.train()
            optimizer.zero_grad()
            x_reconstructed = model(data)
            loss = criterion(data, x_reconstructed)
            loss.backward()
            optimizer.step()
            # Step the scheduler
        scheduler.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]}')
    print("Training complete")

    # sort reconstruction errors
    distances = [criterion(data_tensor[i, :].unsqueeze(0),
                           model(data_tensor)[i, :].unsqueeze(0)).item() for i in range(len(df))]


    sorted_distances = np.sort(distances)

    # Use KneeLocator to find the knee point
    knee = KneeLocator(range(len(sorted_distances)), sorted_distances, curve='convex', direction='increasing')
    treshold = knee.knee_y
    print(f"treshold was set to: {treshold:.4f}")

    outlier_index = [-1 if i > treshold / 2 else 0 for i in distances]
    print(f"percentage of outliers was set to: {-np.sum(outlier_index) / len(outlier_index) * 100: .2f}%")

    return outlier_index


if __name__ == '__main__':
    main()
