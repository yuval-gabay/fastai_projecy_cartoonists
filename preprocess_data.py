import os
from PIL import Image

# --- CONFIGURATION ---
# Define the root directory where your cartoonist folders (tartakovsky, groening, timm) are located
ROOT_DIR = 'imageData'
# Define the target size (square is best). This is the 'pre-sizing' step.
TARGET_SIZE = (460, 460)
# Define the JPEG quality (90 is a good balance of quality and size)
JPEG_QUALITY = 90


# ---------------------

def preprocess_images(root_dir, target_size, jpeg_quality):
    """
    Iterates through subfolders, converts files to RGB, resizes to TARGET_SIZE,
    and saves them as high-quality JPEGs, overwriting the originals.
    """

    # Check if the root directory exists
    if not os.path.exists(root_dir):
        print(f"[ERROR] Root directory not found: {root_dir}")
        print("Please create the 'imageData' folder and put your artist subfolders inside it.")
        return

    print(f"Starting pre-processing in directory: {root_dir}")
    total_processed = 0

    # Get a list of all cartoonist subdirectories
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        # Only process directories (your artist folders)
        if os.path.isdir(subdir_path):
            print(f"\n--- Processing '{subdir}' folder ---")

            # Create a list of files to process to avoid modifying the list while iterating
            files_to_process = [f for f in os.listdir(subdir_path) if
                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

            for filename in files_to_process:
                file_path = os.path.join(subdir_path, filename)

                try:
                    # Open the image
                    img = Image.open(file_path)

                    # Step A: Convert to RGB (Crucial for deep learning models)
                    # This handles transparency (alpha channel) and ensures 3 channels
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Step B: Resize (Resampling minimizes quality loss)
                    # We use Image.Resampling.LANCZOS for high quality downscaling/upscaling
                    img = img.resize(target_size, Image.Resampling.LANCZOS)

                    # Step C: Save (Standardize to JPEG, overwriting or creating a new file)
                    new_filename = os.path.splitext(filename)[0] + '.jpg'
                    new_file_path = os.path.join(subdir_path, new_filename)
                    img.save(new_file_path, 'jpeg', quality=jpeg_quality)

                    # Clean up original file if the format changed (e.g., deleted old PNG)
                    if file_path != new_file_path and os.path.exists(file_path):
                        os.remove(file_path)

                    total_processed += 1

                except Exception as e:
                    print(f"  [ERROR] Could not process {filename}: {e}")

    print(f"\n✅ Pre-processing complete. Total images processed: {total_processed}")


# --- EXECUTE THE FUNCTION ---
if __name__ == "__main__":
    # Note: You must have 'Pillow' installed: pip install Pillow
    preprocess_images(ROOT_DIR, TARGET_SIZE, JPEG_QUALITY)