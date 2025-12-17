import sys
from fastai.vision.all import *

# --- 1. CONFIGURATION ---

# The name of the saved model file
MODEL_FILE = 'cartoon-classifier.pkl'

# Set a default path for a test image, but ideally take it from command line arguments
if len(sys.argv) > 1:
    NEW_IMAGE_PATH = Path(sys.argv[1])
else:
    # Fallback to a default image if no path is provided
    print("[WARNING] No image path provided. Using default test image.")
    NEW_IMAGE_PATH = Path('test_images/sample_test.jpg')  # <-- CHANGE THIS DEFAULT PATH

# --- 2. LOAD THE TRAINED MODEL ---
try:
    learn_inference = load_learner(MODEL_FILE)
    print(f"Model '{MODEL_FILE}' loaded successfully.")
except FileNotFoundError:
    print(f"[ERROR] Model file '{MODEL_FILE}' not found. Did you run the training script and export the model?")
    sys.exit(1)

# --- 3. MAKE PREDICTION ---
if NEW_IMAGE_PATH.exists():
    print(f"\n--- Analyzing drawing: {NEW_IMAGE_PATH.name} ---")

    # The predict method automatically handles necessary pre-processing (like resizing)
    is_artist, index, probs = learn_inference.predict(NEW_IMAGE_PATH)

    # Convert the probabilities to a readable format
    confidence_scores = dict(zip(learn_inference.dls.vocab, [f"{p:.4f}" for p in probs.tolist()]))

    # --- Output the Results ---
    print(f"\n✅ The predicted artist style is **{is_artist}**.")
    print(f"   Confidence: {float(confidence_scores[str(is_artist)]):.2%}")

    print("\nDetailed Probabilities:")
    for artist, score in sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True):
        print(f" - {artist}: {float(score):.2%}")

else:
    print(f"[ERROR] New image file not found at: {NEW_IMAGE_PATH}")