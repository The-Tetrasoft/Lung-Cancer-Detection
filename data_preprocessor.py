import pandas as pd
import os
import shutil
import random
from tqdm import tqdm

# --- USER CONFIGURATION ---
# Root directory of the dataset created by the previous script
FILTERED_DATASET_ROOT = 'D:/Task/Filtered_CXR_Dataset'

# Root directory for the new model-ready dataset
MODEL_DATASET_ROOT = 'D:/Task/CRX_Model_Dataset'

# Target counts for each category to be included in the new dataset.
# These keys must match the folder names inside FILTERED_DATASET_ROOT.
TARGET_CATEGORY_COUNTS = {
    'Mass': 4852,
    'Nodule': 5429,
    'No Finding': 6000
}

# Ratio for splitting data into training and testing sets (e.g., 0.8 for 80% training)
TRAIN_SPLIT_RATIO = 0.8

# Random seed for reproducibility of sampling and splitting
RANDOM_SEED = 42
# --------------------------

def create_model_dataset(filtered_dataset_root, model_dataset_root, target_counts, train_ratio, seed):
    """
    Creates a new dataset for model training and testing from a pre-filtered dataset.
    It samples a specified number of images per category, splits them into
    training and testing sets, copies the images, and generates corresponding metadata CSVs.
    """
    random.seed(seed) # Set random seed for reproducibility

    # Define paths for metadata and output directories
    filtered_metadata_csv = os.path.join(filtered_dataset_root, 'filtered_metadata.csv')
    train_dir = os.path.join(model_dataset_root, 'Training_set')
    test_dir = os.path.join(model_dataset_root, 'Testing_set')

    print(f"Starting creation of CRX_Model_Dataset at: {model_dataset_root}")

    # 1. Load the filtered metadata CSV
    try:
        df_filtered = pd.read_csv(filtered_metadata_csv)
        print(f"✅ Loaded filtered metadata from: {filtered_metadata_csv} ({len(df_filtered)} entries)")
    except FileNotFoundError:
        print(f"❌ Error: filtered_metadata.csv not found at {filtered_metadata_csv}.")
        print("Please ensure the previous script ('lung cancer image organize.py') has been run successfully.")
        return

    # 2. Create main output directories for training and testing sets
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"✅ Created main dataset folders: {train_dir}, {test_dir}")

    train_metadata_rows = [] # To store metadata for the training set
    test_metadata_rows = []  # To store metadata for the testing set
    total_images_processed = 0

    # 3. Process each category (Mass, Nodule, No Finding, Mass_and_Nodule)
    print("\n📦 Processing images for each category...")
    for category_name, target_count in target_counts.items():
        print(f"\n--- Category: {category_name} (Target: {target_count} images) ---")
        source_category_dir = os.path.join(filtered_dataset_root, category_name)

        if not os.path.exists(source_category_dir):
            print(f"⚠️ Warning: Source directory not found for {category_name}: {source_category_dir}. Skipping this category.")
            continue

        # Get all image filenames present in the source category folder
        all_image_filenames = [f for f in os.listdir(source_category_dir) if f.lower().endswith('.png')]
        print(f"  Found {len(all_image_filenames)} images in source folder: {source_category_dir}")

        # Sample the required number of images for this category
        if len(all_image_filenames) < target_count:
            print(f"⚠️ Warning: Not enough images in {category_name} ({len(all_image_filenames)}) to meet target ({target_count}). Using all available.")
            selected_filenames = all_image_filenames
        else:
            selected_filenames = random.sample(all_image_filenames, target_count)
        
        random.shuffle(selected_filenames) # Shuffle the selected images before splitting

        # Split the selected images into training and testing sets for this category
        num_train_cat = int(len(selected_filenames) * train_ratio)
        train_filenames_cat = selected_filenames[:num_train_cat]
        test_filenames_cat = selected_filenames[num_train_cat:]

        print(f"  - Selected {len(selected_filenames)} images for this category.")
        print(f"  - Splitting: {len(train_filenames_cat)} for training, {len(test_filenames_cat)} for testing.")

        # Create subdirectories for the current category within the train/test sets
        dest_train_cat_dir = os.path.join(train_dir, category_name)
        dest_test_cat_dir = os.path.join(test_dir, category_name)
        os.makedirs(dest_train_cat_dir, exist_ok=True)
        os.makedirs(dest_test_cat_dir, exist_ok=True)

        # Copy images and collect metadata for the training set
        for filename in tqdm(train_filenames_cat, desc=f"  Copying {category_name} (Train)"):
            src_path = os.path.join(source_category_dir, filename)
            dest_path = os.path.join(dest_train_cat_dir, filename)
            shutil.copy2(src_path, dest_path) # Use copy2 to preserve metadata like timestamps
            
            # Find the corresponding metadata row from the loaded filtered_metadata.csv
            row_data = df_filtered[df_filtered['Image Index'] == filename].iloc[0].to_dict()
            train_metadata_rows.append(row_data)
            total_images_processed += 1

        # Copy images and collect metadata for the testing set
        for filename in tqdm(test_filenames_cat, desc=f"  Copying {category_name} (Test)"):
            src_path = os.path.join(source_category_dir, filename)
            dest_path = os.path.join(dest_test_cat_dir, filename)
            shutil.copy2(src_path, dest_path)
            
            # Find the corresponding metadata row
            row_data = df_filtered[df_filtered['Image Index'] == filename].iloc[0].to_dict()
            test_metadata_rows.append(row_data)
            total_images_processed += 1

    # 4. Create and save metadata CSVs for the training and testing sets
    print("\n📝 Creating metadata CSV files for training and testing sets...")
    
    if train_metadata_rows:
        df_train = pd.DataFrame(train_metadata_rows)
        df_train = df_train[df_filtered.columns] # Ensure columns are in the same order as original
        train_csv_path = os.path.join(train_dir, 'training_metadata.csv')
        df_train.to_csv(train_csv_path, index=False)
        print(f"✅ Training metadata saved to: {train_csv_path} ({len(df_train)} entries)")
    else:
        print("⚠️ No training images processed, skipping training metadata CSV creation.")

    if test_metadata_rows:
        df_test = pd.DataFrame(test_metadata_rows)
        df_test = df_test[df_filtered.columns] # Ensure columns are in the same order as original
        test_csv_path = os.path.join(test_dir, 'testing_metadata.csv')
        df_test.to_csv(test_csv_path, index=False)
        print(f"✅ Testing metadata saved to: {test_csv_path} ({len(df_test)} entries)")
    else:
        print("⚠️ No testing images processed, skipping testing metadata CSV creation.")

    print("\n--- Summary ---")
    print(f"🎉 Dataset creation complete! Total images processed: {total_images_processed}")
    print(f"Training set images: {len(train_metadata_rows)}")
    print(f"Testing set images: {len(test_metadata_rows)}")
    print(f"New dataset root: {model_dataset_root}")

# Run the function when the script is executed
if __name__ == "__main__":
    create_model_dataset(
        FILTERED_DATASET_ROOT,
        MODEL_DATASET_ROOT,
        TARGET_CATEGORY_COUNTS,
        TRAIN_SPLIT_RATIO,
        RANDOM_SEED
    )