import gradio as gr
import sys
import os

# Add the project root to the Python path to allow importing from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import load_model, predict_genre, GENRES

# --- Model Loading ---
model = None
model_error = None

try:
    model = load_model()
    print("Model loaded successfully.")
except FileNotFoundError:
    model_error = "Model file not found. Please train the model first by running `python src/train.py`."
    print(model_error)
except Exception as e:
    model_error = f"An error occurred while loading the model: {e}"
    print(model_error)

# --- UI Functions ---
def classify_music(audio_filepath):
    """Wrapper function for Gradio to handle audio input and return predictions."""
    if model is None:
        return "Model is not loaded. Cannot perform classification.", {genre: 0 for genre in GENRES}
    
    if audio_filepath is None:
        return "Please upload an audio file.", {genre: 0 for genre in GENRES}
    
    try:
        predicted_genre, confidences = predict_genre(model, audio_filepath)
        return predicted_genre, confidences
    except Exception as e:
        return f"An error occurred during prediction: {e}", {genre: 0 for genre in GENRES}

# --- Gradio Interface ---
with gr.Blocks() as interface:
    gr.Markdown("<h1>Music Genre Classifier</h1>")
    gr.Markdown("Upload a short audio clip (.wav) and the model will predict its genre.")

    if model_error:
        gr.Markdown(f"<p style='color:red;font-weight:bold;'>Error: {model_error}</p>")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio File")
        
        with gr.Column():
            predicted_genre_output = gr.Textbox(label="Predicted Genre")
            confidence_output = gr.Label(num_top_classes=10, label="Confidence Scores")

    # The button is disabled if the model failed to load
    submit_btn = gr.Button("Classify", interactive=(model is not None))
    submit_btn.click(
        fn=classify_music,
        inputs=audio_input,
        outputs=[predicted_genre_output, confidence_output]
    )

# --- Launch the App ---
if __name__ == "__main__":
    interface.launch()
