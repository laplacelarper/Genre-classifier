import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from torch_audiomentations import Compose
from torch_audiomentations.augmentations.colored_noise import AddColoredNoise
from torch_audiomentations.augmentations.pitch_shift import PitchShift

class GenreDataset(Dataset):
    """Custom PyTorch Dataset for the GTZAN genre classification dataset."""

    def __init__(self, data_dir, transform=None, target_sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512, fixed_length=128, training=True, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fixed_length = fixed_length
        self.training = training
        self.augment = augment
        self.samples = self._load_samples()
        self.genres = sorted(os.listdir(data_dir))

        if self.augment:
            # Note: TimeStretch is not available in this version of torch-audiomentations
            self.augmentation_pipeline = Compose([
                AddColoredNoise(min_snr_in_db=3.0, max_snr_in_db=10.0, p=0.5, sample_rate=self.target_sample_rate),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5, sample_rate=self.target_sample_rate)
            ])

    def _load_samples(self):
        samples = []
        for genre in os.listdir(self.data_dir):
            genre_dir = os.path.join(self.data_dir, genre)
            if os.path.isdir(genre_dir):
                for filename in os.listdir(genre_dir):
                    if filename.endswith('.wav'):
                        filepath = os.path.join(genre_dir, filename)
                        samples.append((filepath, genre))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, genre_name = self.samples[index]
        genre_index = self.genres.index(genre_name)

        try:
            # Load audio file and convert to a tensor for augmentation
            signal, sr = librosa.load(filepath, sr=self.target_sample_rate, duration=5)
            signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        except Exception as e:
            print(f"Warning: Skipping corrupted file {filepath}: {e}")
            return None, None # Return None for corrupted files

        # Apply augmentations if enabled
        if self.augment:
            signal = self.augmentation_pipeline(samples=signal, sample_rate=self.target_sample_rate).squeeze(0)

        # Extract Mel Spectrogram
        signal_np = signal.numpy() # Convert back to numpy for librosa
        mel_spectrogram = librosa.feature.melspectrogram(y=signal_np,
                                                         sr=self.target_sample_rate,
                                                         n_fft=self.n_fft,
                                                         hop_length=self.hop_length,
                                                         n_mels=self.n_mels)
        
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Pad or truncate the spectrogram to a fixed length
        if self.training:
            # Random crop for training
            if mel_spectrogram.shape[1] > self.fixed_length:
                start = np.random.randint(0, mel_spectrogram.shape[1] - self.fixed_length)
                mel_spectrogram = mel_spectrogram[:, start:start + self.fixed_length]
            else:
                mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, self.fixed_length - mel_spectrogram.shape[1])), mode='constant')
        else:
            # Center crop for validation/testing
            if mel_spectrogram.shape[1] > self.fixed_length:
                start = (mel_spectrogram.shape[1] - self.fixed_length) // 2
                mel_spectrogram = mel_spectrogram[:, start:start + self.fixed_length]
            else:
                mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, self.fixed_length - mel_spectrogram.shape[1])), mode='constant')

        # Convert to tensor and add channel dimension
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0)

        # Apply normalization if a transform is provided
        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        return mel_spectrogram, genre_index
