import torch
import torch
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import GenreDataset
from src.model import GenreClassifierCNN

def collate_fn(batch):
    """Custom collate function to filter out None samples."""
    # Filter out None values returned by the dataset's __getitem__ for corrupted files
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None # Return None if the whole batch was corrupted
    return torch.utils.data.dataloader.default_collate(batch)

import os
import kagglehub

# --- Configuration ---
MODEL_SAVE_PATH = 'models/model.pth'
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8

def train():
    """Main training loop for the genre classifier."""
    # --- 1. Download and Prepare Dataset ---
    print("Downloading GTZAN dataset from Kaggle...")
    try:
        # The path will point to the directory containing the dataset files
        download_path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
        
        # The actual genre folders are in a subdirectory, let's find it
        data_dir = os.path.join(download_path, 'Data', 'genres_original')
        if not os.path.isdir(data_dir):
            # Fallback for different structures
            data_dir = os.path.join(download_path, 'genres_original')
            if not os.path.isdir(data_dir):
                raise FileNotFoundError("Could not find the 'genres_original' directory in the downloaded dataset.")

        print(f"Dataset downloaded and located at: {data_dir}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have configured your Kaggle API credentials.")
        print("See: https://www.kaggle.com/docs/api")
        return # Exit if download fails

    

    # --- 2. Load Dataset ---
    print("Loading and preprocessing data...")
    # Create separate datasets for training and testing to apply different transforms
    train_dataset_for_stats = GenreDataset(data_dir=data_dir, training=True)
    
    # --- Calculate Mean and Std for Normalization ---
    print("Calculating mean and std of the training set for normalization...")
    # Use a subset for faster calculation if the dataset is very large
    # For GTZAN, using the full training set is fine.
    loader_for_stats = DataLoader(train_dataset_for_stats, batch_size=BATCH_SIZE, num_workers=2, collate_fn=collate_fn)
    
    mean = 0.
    std = 0.
    num_samples = 0
    for inputs, _ in loader_for_stats:
        batch_samples = inputs.size(0)
        inputs = inputs.view(batch_samples, inputs.size(1), -1)
        mean += inputs.mean(2).sum(0)
        std += inputs.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std: {std}")

    # --- 2. Create Datasets with Normalization ---
    data_transform = transforms.Compose([
        transforms.Normalize(mean=[mean.item()], std=[std.item()])
    ])

    # Create two datasets: one for training (with augmentations) and one for testing (without)
    train_dataset_augmented = GenreDataset(data_dir=data_dir, transform=data_transform, training=True, augment=True)
    test_dataset_no_aug = GenreDataset(data_dir=data_dir, transform=data_transform, training=False, augment=False)

    # --- 2. Split Dataset ---
    # Ensure the split is the same for both datasets by using a generator with a fixed seed.
    # We create subsets of indices and apply them to our two different dataset objects.
    num_samples = len(train_dataset_augmented)
    train_size = int(TRAIN_SPLIT * num_samples)
    test_size = num_samples - train_size
    
    # Generate indices for the split
    indices = list(range(num_samples))
    # Note: We don't need to shuffle here because the DataLoader will shuffle the training set.
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create subsets based on the indices
    train_dataset = torch.utils.data.Subset(train_dataset_augmented, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset_no_aug, test_indices)
    


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = GenreClassifierCNN(num_genres=len(train_dataset.dataset.genres))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device}")

    # --- 4. Training Loop ---
    best_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Skip batch if it's empty after filtering
            if inputs is None or labels is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

        # --- 5. Validation Loop ---
        model.eval()
        correct = 0
        total = 0
        # On the last epoch, collect data for confusion matrix
        if epoch == NUM_EPOCHS - 1:
            all_labels_cm = []
            all_predictions_cm = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                if inputs is None:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # Collect labels and predictions for the last epoch
                if epoch == NUM_EPOCHS - 1:
                    all_labels_cm.extend(labels.cpu().numpy())
                    all_predictions_cm.extend(predicted.cpu().numpy())
        
        accuracy = 100 * correct / total
        print(f'Accuracy on test set: {accuracy:.2f} %')

        # --- 6. Save Best Model ---
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")

    # --- 7. Generate and Save Confusion Matrix ---
    # This uses the labels and predictions from the last epoch's evaluation
    print("Generating confusion matrix...")
    cm = confusion_matrix(all_labels_cm, all_predictions_cm)
    genres = full_dataset.genres

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=genres, yticklabels=genres, cmap='Blues')
    plt.title('Confusion Matrix for Final Epoch')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Ensure the results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to results/confusion_matrix.png")

    print("Finished Training")

if __name__ == '__main__':
    train()
