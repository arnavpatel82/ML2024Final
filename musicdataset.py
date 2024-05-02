import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset
import numpy as np

class MusicDataset(Dataset):
    def __init__(self,audio_dir_path, labels_path, mel_spectrogram):
        self.labels = self.read_file_to_numpy_array(labels_path).astype(int)
        self.audio_dir_path = audio_dir_path
        self.mel_spectrogram = mel_spectrogram
        
        
    def __getitem__(self, index):
        mp3_path = f'{self.audio_dir_path}/{index}.mp3'
        spectrogram = self.audio_to_spectrogram(mp3_path)
        label = self.labels[index]
        return spectrogram, int(label)
        
        
    def read_file_to_numpy_array(self, filename):
        with open(filename, 'r') as file:
            lines = file.read().splitlines()
            lines_array = np.array(lines)
        return lines_array
    
    
    def audio_to_spectrogram(self, file_path):
        # Load audio file
        waveform, _ = torchaudio.load(file_path)
        # Apply transform to waveform
        mel_spec = self.mel_spectrogram(waveform)
        return mel_spec
  
    
if __name__ == "__main__":
    labels_path = 'Data/train_label.txt'
    training_data = 'Data/train_mp3s'
    mel_spectrogram = MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
    mds = MusicDataset(training_data,labels_path,mel_spectrogram)
    spect, label = mds[2]
    print(spect.shape)
    
    
    
    

        