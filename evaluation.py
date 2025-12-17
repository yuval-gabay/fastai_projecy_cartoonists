from fastai.vision.all import *
import matplotlib.pyplot as plt
import warnings

# --- CONFIGURATION ---
MODEL_FILE = 'cartoon-classifier.pkl'
DATA_PATH = Path('imageData')  # Folder containing your artist subfolders


def run_evaluation():
    print(f"Step 1: Loading model '{MODEL_FILE}'...")
    try:
        # 1. Load the model
        learn = load_learner(MODEL_FILE)

        # 2. Re-attach the data structure
        # This ensures the model knows the labels and validation set
        print(f"Step 2: Checking for images in '{DATA_PATH}'...")
        if not DATA_PATH.exists():
            print(f"[ERROR]: Folder '{DATA_PATH}' not found. Please check your directory.")
            return

        dls = ImageDataLoaders.from_folder(DATA_PATH, valid_pct=0.2, item_tfms=Resize(224))
        learn.dls = dls
        print(f"Success: Model and Data loaded. Vocabulary: {dls.vocab}")

        # 3. Create Interpretation (This calculates predictions on validation set)
        print("Step 3: Calculating predictions (Interpretation)...")
        interp = ClassificationInterpretation.from_learner(learn)

        # 4. Generate and Save Confusion Matrix
        print("Step 4: Generating Confusion Matrix...")
        interp.plot_confusion_matrix(figsize=(8, 8))
        plt.title("Confusion Matrix")
        plt.savefig('confusion_matrix.png')  # Saves to your project folder
        plt.close()
        print("--- Saved: 'confusion_matrix.png' ---")

        # 5. Generate and Save Top Losses
        print("Step 5: Generating Top Losses plot...")
        interp.plot_top_losses(k=9, figsize=(10, 10))
        plt.savefig('top_losses.png')  # Saves to your project folder
        plt.close()
        print("--- Saved: 'top_losses.png' ---")

        print("\nEvaluation Complete! Check your project folder for the .png files.")

    except Exception as e:
        print(f"[FATAL ERROR]: {e}")


if __name__ == "__main__":
    run_evaluation()