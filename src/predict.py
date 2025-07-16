import torch
import librosa
import numpy as np
from src.model import GenreClassifierCNN

# --- Configuration ---
MODEL_PATH = 'models/model.pth'
TARGET_SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FIXED_LENGTH = 128
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_model(model_path=MODEL_PATH):
    """Loads the trained model from the specified path."""
    model = GenreClassifierCNN(num_genres=len(GENRES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_genre(model, file_path):
    """Predict the genre of a single audio file using the provided model."""
    # 1. Load and preprocess the audio file
    signal, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
    
    # 2. Extract mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal, 
        sr=sr, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # 3. Resize to fixed length
    if mel_spectrogram_db.shape[1] > FIXED_LENGTH:
        mel_spectrogram_db = mel_spectrogram_db[:, :FIXED_LENGTH]
    else:
        pad_width = FIXED_LENGTH - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # 4. Add batch and channel dimensions
    spectrogram_tensor = torch.tensor(mel_spectrogram_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 5. Make prediction
    with torch.no_grad():
        output = model(spectrogram_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        _, predicted_idx = torch.max(output, 1)
        predicted_genre = GENRES[predicted_idx.item()]
    
    # 6. Create a dictionary of genre probabilities
    confidences = {genre: prob.item() for genre, prob in zip(GENRES, probabilities)}
    
    return predicted_genre, confidences

if __name__ == '__main__':
    # Example usage: This script can be run directly to test a single audio file.
    # It requires a trained model at models/model.pth.
    try:
        # Load the model first
        model = load_model()
        print("Model loaded successfully for standalone prediction.")

        # Create a dummy audio file for testing
        import soundfile as sf
        dummy_audio = np.random.randn(TARGET_SAMPLE_RATE * 5)  # 5 seconds of noise
        dummy_path = 'dummy_audio.wav'
        sf.write(dummy_path, dummy_audio, TARGET_SAMPLE_RATE)
        print(f"Created a dummy audio file at: {dummy_path}")

        # Predict the genre
        predicted_genre, confidences = predict_genre(model, dummy_path)
        
        print(f"\nPredicted Genre: {predicted_genre}")
        
        # Format and print the confidences
        formatted_confidences = {genre: f"{prob*100:.2f}%" for genre, prob in confidences.items()}
        print("Confidences:", formatted_confidences)

    except FileNotFoundError:
        print(f"\nError: Model not found at {MODEL_PATH}. Please train the model first by running 'python src/train.py'.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
