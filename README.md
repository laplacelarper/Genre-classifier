# Music Genre Classifier

This project is a deep learning-based music genre classifier built with PyTorch. It uses mel spectrograms generated from audio clips to classify them into one of 10 genres. The application is deployed with an interactive web UI using Gradio.

## 🎼 Tech Stack

- **Language**: Python 3.13
- **Deep Learning**: PyTorch
- **Audio Processing**: Librosa
- **Data**: GTZAN Dataset
- **UI/Visualization**: Gradio

## 📁 Folder Structure

```
music_genre_classifier/
├── app/
│   └── app.py               # Gradio UI
├── data/
│   ├── raw/                 # Raw audio files (.wav)
│   └── processed/           # (Optional) Processed data
├── models/
│   └── model.pth            # Trained PyTorch model
├── notebooks/
│   └── training.ipynb       # (Optional) Training notebook
├── src/
│   ├── dataset.py           # Custom PyTorch dataset
│   ├── model.py             # CNN architecture
│   ├── train.py             # Training script
│   └── predict.py           # Inference script
├── venv/                    # Python virtual environment
├── requirements.txt
└── README.md
```

## ⚙️ Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd music_genre_classifier
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🎵 Dataset (GTZAN)

The project now automatically downloads the GTZAN dataset using `kagglehub`. This requires you to have your Kaggle API credentials configured on your system.

**To set up your Kaggle credentials:**

1.  **Create a Kaggle Account**: If you don't have one, sign up at [kaggle.com](https://www.kaggle.com).
2.  **Generate an API Token**: Go to your Kaggle account settings, find the "API" section, and click "Create New API Token". This will download a `kaggle.json` file.
3.  **Place the Token**: Move the `kaggle.json` file to the required location:
    - **macOS/Linux**: `~/.kaggle/kaggle.json`
    - **Windows**: `C:\Users\<Your-Username>\.kaggle\kaggle.json`

    You may need to create the `.kaggle` directory if it doesn't exist.

## 🏋️ Training the Model

To train the model, simply run the training script from the project root:

```bash
python src/train.py
```

The script will now automatically:
- Download the GTZAN dataset using `kagglehub`.
- Preprocess the audio into mel spectrograms.
- Split the data into training and testing sets (80/20).
- Train the CNN model for 20 epochs.
- Save the model with the best validation accuracy to `models/model.pth`.

## 🚀 Running the Application

Once the model is trained and `model.pth` exists, you can launch the Gradio web application:

```bash
python app/app.py
```

This will start a local web server. Open the provided URL in your browser to access the interactive UI. You can upload a `.wav` file or use one of the provided examples to see the model's prediction.

### Sample UI

*(A screenshot of the Gradio app would go here. You can generate one after running the app.)*

## 🧠 Model Architecture

The model is a Convolutional Neural Network (CNN) designed for image-like data (in this case, mel spectrograms). The architecture is as follows:

1.  **Input**: Mel Spectrogram (1x128x128)
2.  **Convolutional Block 1**:
    - `Conv2d` (1 -> 16 channels, kernel=3)
    - `BatchNorm2d`
    - `ReLU`
    - `MaxPool2d`
3.  **Convolutional Block 2**:
    - `Conv2d` (16 -> 32 channels, kernel=3)
    - `BatchNorm2d`
    - `ReLU`
    - `MaxPool2d`
4.  **Convolutional Block 3**:
    - `Conv2d` (32 -> 64 channels, kernel=3)
    - `ReLU`
    - `AdaptiveAvgPool2d`
5.  **Classifier Head**:
    - `Flatten`
    - `Linear` (64 -> 10 output classes)

