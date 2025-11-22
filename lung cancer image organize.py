import os
import shutil
import pandas as pd
import random
from tqdm import tqdm

def create_small_dataset(source_dir, dest_dir, num_images_per_class=500):
    """
    Creates a smaller dataset by sampling a specified number of images from each class
    in the training and testing sets of a source directory.

    Args:
        source_dir (str): The path to the source dataset (e.g., 'CRX_Model_Dataset').
        dest_dir (str): The path to the destination directory for the small dataset (e.g., 'Small_Dataset').
        num_images_per_class (int): The number of images to sample from each class.
    """
    print(f"Creating small dataset at: {dest_dir}")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for set_name in ['Training_set', 'Testing_set']:
        source_set_dir = os.path.join(source_dir, set_name)
        dest_set_dir = os.path.join(dest_dir, set_name)
        
        print(f"\nProcessing subset: {set_name}")
        if not os.path.exists(dest_set_dir):
            os.makedirs(dest_set_dir)

        # Read original metadata
        source_metadata_path = os.path.join(source_set_dir, f'{set_name.lower().replace("_set", "")}_metadata.csv')
        if not os.path.exists(source_metadata_path):
            print(f"  - Warning: Metadata file not found at {source_metadata_path}. Skipping metadata processing for this subset.")
            original_df = None
        else:
            original_df = pd.read_csv(source_metadata_path)
        
        all_selected_images = []

        # Define classes explicitly to ensure order and inclusion
        classes = ['Mass', 'Nodule', 'No Finding']

        for class_name in classes:
            source_class_dir = os.path.join(source_set_dir, class_name)
            dest_class_dir = os.path.join(dest_set_dir, class_name)
            
            print(f"  - Processing class: {class_name}")
            if not os.path.exists(dest_class_dir):
                os.makedirs(dest_class_dir)

            if not os.path.exists(source_class_dir):
                print(f"    - Warning: Source class directory not found: {source_class_dir}. Skipping.")
                continue

            images = [f for f in os.listdir(source_class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)
            
            selected_images = images[:num_images_per_class]
            all_selected_images.extend(selected_images)

            print(f"    - Selecting {len(selected_images)} of {len(images)} images.")

            for image in tqdm(selected_images, desc=f"Copying {class_name} images"):
                shutil.copy(os.path.join(source_class_dir, image), os.path.join(dest_class_dir, image))
        
        # Filter and save new metadata
        if original_df is not None and all_selected_images:
            # The column with image names in 'training_metadata.csv' is 'Image Index'
            image_index_col = 'Image Index'
            if image_index_col in original_df.columns:
                filtered_df = original_df[original_df[image_index_col].isin(all_selected_images)]
                
                # Create a new name for the metadata file in the small dataset
                new_metadata_filename = f'{set_name.lower().replace("_set", "")}_metadata.csv'
                dest_metadata_path = os.path.join(dest_set_dir, new_metadata_filename)
                
                filtered_df.to_csv(dest_metadata_path, index=False)
                print(f"  - Saved new metadata for {set_name} at {dest_metadata_path} with {len(filtered_df)} entries.")
            else:
                print(f"  - Warning: '{image_index_col}' column not found in {source_metadata_path}. Cannot filter metadata.")

if __name__ == '__main__':
    # Using absolute paths for clarity and safety
    SOURCE_DATASET = 'd:\\Task\\CRX_Model_Dataset'
    DEST_DATASET = 'd:\\Task\\Small_Dataset'
    IMAGES_PER_CLASS = 500

    # Ensure the script is executable and has the necessary libraries
    print("--- Small Dataset Creation Script ---")
    print("This script will create a smaller, balanced dataset from the original.")
    print(f"Source: {SOURCE_DATASET}")
    print(f"Destination: {DEST_DATASET}")
    print(f"Images per class: {IMAGES_PER_CLASS}")
    print("Ensure you have 'pandas' and 'tqdm' installed (`pip install pandas tqdm`).")
    
    create_small_dataset(SOURCE_DATASET, DEST_DATASET, IMAGES_PER_CLASS)
    
    print("\n--- Script Finished ---")
    print(f"The new dataset is located at: {DEST_DATASET}")
