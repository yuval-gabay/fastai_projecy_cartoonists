import sys
from fastai.vision.all import *
from pathlib import Path
import os
import warnings
from PIL import ImageFile

# --- 1. CONFIGURATION ---

# The name of the saved model file (must be in the same folder as this script)
MODEL_FILE = 'cartoon-classifier.pkl'

# --- FIXED FOLDER PATH ---
# The script will always look inside this folder in the current project directory.
TEST_FOLDER_NAME = 'testImages'
# -------------------------

# Increase PIL's limit to handle large/complex images if needed
ImageFile.LOAD_TRUNCATED_IMAGES = True


def predict_style(image_path, learn):
    """
    Loads a single image, makes a prediction, and prints the result.
    """
    try:
        # Load the image using fastai's method
        img = PILImage.create(image_path)
    except FileNotFoundError:
        print(f"[ERROR] Image file not found at: {image_path}")
        return
    except Exception as e:
        print(f"[ERROR] Could not open or process image {os.path.basename(image_path)}: {e}")
        return

    # Suppress the UserWarning about the deprecated cnn_learner when predicting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Get the prediction and the probabilities (raw output)
        pred, pred_idx, probs = learn.predict(img)

    # Convert probabilities to a list of formatted strings (e.g., "95.12%")
    probs_list = [f"{p.item() * 100:.2f}%" for p in probs]

    # Map the class labels to the probability values
    results = dict(zip(learn.dls.vocab, probs_list))

    # Get the confidence of the top prediction
    confidence = float(probs_list[pred_idx.item()].replace('%', ''))

    print(f"\n--- Analyzing file: {os.path.basename(image_path)} ---")
    print(f"✅ Predicted Style: **{pred}**")
    print(f"   Confidence: {confidence:.2f}%")
    print(f"   Detailed Probabilities: {results}")


def process_folder():
    """
    Loads the model, finds all images in the fixed folder, and calls predict_style for each.
    """
    # 1. Load Model
    print(f"Loading model '{MODEL_FILE}'...")
    try:
        # Suppress the load_learner pickle security warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learn = load_learner(MODEL_FILE)
            print("Model loaded successfully.")
    except Exception as e:
        print(f"[FATAL ERROR] Could not load model '{MODEL_FILE}'. Ensure it is in the same directory.")
        print(f"Details: {e}")
        return

    # 2. Define Folder Path
    # Path.cwd() gives the current project directory, then we join the test folder name.
    folder_path = Path.cwd() / TEST_FOLDER_NAME

    if not folder_path.is_dir():
        print(f"[ERROR] The required test directory was not found: {folder_path.resolve()}")
        print("Please ensure you have a folder named 'testImages' in your project root.")
        return

    print(f"Starting batch prediction on folder: {folder_path.resolve()}")

    # 3. Find Image Files
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']

    # Use glob to find all files matching the extensions in the folder
    image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print("[WARNING] No supported image files found in the 'testImages' directory.")
        print("Please ensure your image files are one of the following types: .png, .jpg, .jpeg, .webp")
        return

    # 4. Process Images
    for i, image_file in enumerate(image_files):
        print(f"\n--- Processing Image {i + 1}/{len(image_files)} ---")
        predict_style(image_file, learn)

    print("\n--- BATCH PREDICTION COMPLETE ---")


# --- EXECUTION ---
if __name__ == "__main__":
    process_folder()