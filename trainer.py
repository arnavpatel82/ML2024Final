import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset

from musicdataset import MusicDataset
from cnn import CNNNetwork


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

LABELS_PATH = "Data/train_label.txt"
TRAINING_DATA_PATH = "Data/train_mp3s"


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser):
    for input, target in data_loader:

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    mel_spectrogram = MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
    mds = MusicDataset(TRAINING_DATA_PATH,LABELS_PATH, mel_spectrogram)
    
    train_dataloader = create_data_loader(mds, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork()
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "/model")