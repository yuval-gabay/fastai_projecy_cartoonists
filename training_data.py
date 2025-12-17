from fastai.vision.all import *

# --- 1. CONFIGURATION ---

# Define the model architecture and training parameters
MODEL_ARCH = resnet34
NUM_EPOCHS = 6#before its was 4
MODEL_FILE = 'cartoon-classifier.pkl'

# IMPORTANT: This path is set explicitly to your local data folder.
# Ensure this folder contains the clean subdirectories: 'tartakovsky', 'timm', 'pendleton'
# NOTE: Using a raw string (r'...') is essential for Windows paths.
DATA_PATH = Path(r'C:\Users\Surface\PycharmProjects\PythonProject2\imageData')


def train_model():
    """
    Sets up the DataLoaders, creates the ResNet34 model, fine-tunes it,
    saves the final model, and prints the evaluation results.
    """

    print(f"Loading data from: {DATA_PATH}")

    # --- 2. DATA BLOCK AND DATALOADERS SETUP ---
    print("Setting up DataLoaders...")

    # The DataBlock defines how to get and prepare the data
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        # Gets the label (e.g., 'timm', 'pendleton') from the folder name
        get_y=parent_label,
        item_tfms=Resize(460),  # Pre-sizing for cropping
        batch_tfms=aug_transforms(
            size=224,  # Final size seen by the model
            min_scale=0.6,  # Random zoom/crop control
            max_warp=0.2,
            max_rotate=25.0
        )
    )

    try:
        # Create the DataLoaders using the path and DataBlock
        dls = dblock.dataloaders(DATA_PATH, bs=32)
    except ValueError as e:
        print(f"[ERROR] Could not create DataLoaders. Check the path and folder structure.")
        print(f"Details: {e}")
        return

    print(f"Data ready! Found {len(dls.train_ds)} training images and {len(dls.valid_ds)} validation images.")

    # --- 3. MODEL CREATION (Transfer Learning) ---
    learn = cnn_learner(dls, MODEL_ARCH, metrics=error_rate)
    print(f"Model created using {MODEL_ARCH.__name__}...")

    # --- 4. MODEL TRAINING (Fine-Tuning) ---
    print(f"Starting fine-tuning for {NUM_EPOCHS} epochs...")
    learn.fine_tune(NUM_EPOCHS)
    print("\n✨ Training complete! ✨")

    # --- 5. SAVE THE MODEL (NEW ADDITION) ---
    learn.export(MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")

    # --- 6. INTERPRETATION AND EVALUATION ---
    print("\n--- Model Evaluation ---")
    results = learn.validate()
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Error Rate (Incorrect Predictions): {results[1].item():.4f}")
    print(f"Accuracy: {1 - results[1].item():.4f}")

    # Optional: You can uncomment these lines to see the Confusion Matrix and Top Losses
    # interp = ClassificationInterpretation.from_learner(learn)
    # interp.plot_confusion_matrix(figsize=(7, 7), title="Confusion Matrix")
    # interp.plot_top_losses(9, figsize=(15, 15))


# --- EXECUTION ENTRY POINT (NEW ADDITION) ---
# This allows the IDE/command line to run the train_model function directly.
if __name__ == "__main__":
    # Ensure you have run preprocess_data.py first!
    train_model()