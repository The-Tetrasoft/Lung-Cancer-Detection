import os
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
from dataOgmentation import (
    get_augmentations,
    test_single_transformation,
    save_augmented_samples,
    visualize_and_adjust,
    log_augmentations_to_tensorboard
)

def test_augmentations():
    """
    Test all augmentation functionality
    """
    # Test configuration
    sample_images_dir = 'd:/Task/Filtered_CXR_Dataset'
    output_dir = 'd:/Task/augmentation_tests'
    
    # Sample images for each class
    sample_images = {
        'Mass': os.path.join(sample_images_dir, 'Mass', '00000012_000.png'),
        'Nodule': os.path.join(sample_images_dir, 'Nodule', '00000013_022.png'),
        'No Finding': os.path.join(sample_images_dir, 'No Finding', '00000013_023.png')
    }

    # Intensity controls for testing
    intensity_controls = {
        'Mass': {
            'rotation': 15,
            'zoom': (0.85, 1.15),
            'perspective': 0.15,
            'noise_std': 0.1,
            'erase_scale': (0.02, 0.1)
        },
        'Nodule': {
            'rotation': 5,
            'zoom': (0.95, 1.05),
            'perspective': 0.05,
            'noise_std': 0.05,
            'erase_scale': (0.01, 0.03)
        },
        'No Finding': {
            'rotation': 20,
            'zoom': (0.8, 1.2),
            'perspective': 0.2,
            'noise_std': 0.15,
            'erase_scale': (0.02, 0.15)
        }
    }

    # Create TensorBoard writer
    writer = SummaryWriter('runs/augmentation_test')

    try:
        # 1. Test individual transformations
        print("\n=== Testing Individual Transformations ===")
        for class_name, image_path in sample_images.items():
            if os.path.exists(image_path):
                print(f"\nTesting {class_name}...")
                class_output_dir = os.path.join(output_dir, 'individual', class_name)
                test_single_transformation(
                    image_path, 
                    class_output_dir,
                    intensity_controls[class_name]
                )

        # 2. Test full augmentation pipeline
        print("\n=== Testing Full Augmentation Pipeline ===")
        for class_name, image_path in sample_images.items():
            if os.path.exists(image_path):
                print(f"\nTesting {class_name}...")
                class_output_dir = os.path.join(output_dir, 'full_pipeline', class_name)
                save_augmented_samples(
                    image_path,
                    class_output_dir,
                    class_name,
                    intensity_controls[class_name],
                    num_samples=5
                )

        # 3. Test TensorBoard logging
        print("\n=== Testing TensorBoard Logging ===")
        for class_name, image_path in sample_images.items():
            if os.path.exists(image_path):
                print(f"\nLogging {class_name}...")
                log_augmentations_to_tensorboard(
                    writer,
                    f'Augmentations/{class_name}',
                    image_path,
                    class_name,
                    intensity_controls[class_name]
                )

        # 4. Test interactive visualization
        print("\n=== Testing Interactive Visualization ===")
        for class_name, image_path in sample_images.items():
            if os.path.exists(image_path):
                print(f"\nVisualizing {class_name}...")
                augmented = visualize_and_adjust(
                    image_path,
                    class_name,
                    intensity_controls[class_name]
                )
                # Save the visualization
                viz_dir = os.path.join(output_dir, 'interactive')
                os.makedirs(viz_dir, exist_ok=True)
                augmented.save(os.path.join(viz_dir, f'{class_name}_interactive.png'))

    except Exception as e:
        print(f"Error during testing: {str(e)}")
    finally:
        writer.close()

    print("\n=== Testing Complete ===")
    print(f"Results saved in: {output_dir}")
    print("TensorBoard logs saved in: runs/augmentation_test")
    print("To view TensorBoard, run: tensorboard --logdir=runs")

if __name__ == "__main__":
    test_augmentations()