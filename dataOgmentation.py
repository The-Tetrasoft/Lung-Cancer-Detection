import os
import shutil
import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms

# --- Local Imports ---
from augmentation_switch import AUGMENTATION_SWITCHES
from torch.utils.tensorboard import SummaryWriter
import random
import torchvision

# --- Custom Transform for Gaussian Noise ---
class AddGaussianNoise(object):
    """Adds Gaussian noise to a PIL image."""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

# --- Grayscale and Histogram Equalization Transforms ---
class GrayscaleTo3Channel(object):
    """Converts a grayscale PIL image to a 3-channel PIL image."""
    def __call__(self, img):
        return img.convert('RGB')

class HistogramEqualization(object):
    """Applies histogram equalization to a grayscale PIL image."""
    def __call__(self, img):
        # Convert to grayscale if it's not
        if img.mode != 'L':
            img = img.convert('L')
        return ImageOps.equalize(img)

# --- Main Augmentation Function ---
def get_augmentations(class_name, intensity_controls):
    """
    Returns a composition of transforms based on the class name and intensity controls.
    """
    # Default intensities, can be overridden by intensity_controls
    defaults = {
        'rotation': 10, 'zoom': (0.9, 1.1), 'perspective': 0.1,
        'noise_std': 0.05
    }
    
    # Class-specific adjustments
    if class_name == 'Mass':
        defaults.update({'rotation': 15, 'zoom': (0.85, 1.15), 'perspective': 0.15})
    elif class_name == 'Nodule':
        defaults.update({'rotation': 5, 'zoom': (0.95, 1.05), 'perspective': 0.05})
    elif class_name == 'No Finding':
        defaults.update({'rotation': 20, 'zoom': (0.8, 1.2), 'perspective': 0.2})

    # Apply user-defined intensities
    config = {**defaults, **intensity_controls}

    # Define transformations
    transform_list = [
        HistogramEqualization(),
        GrayscaleTo3Channel()
    ]

    if AUGMENTATION_SWITCHES.get('use_horizontal_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if AUGMENTATION_SWITCHES.get('use_rotation', False):
        transform_list.append(transforms.RandomRotation(config['rotation']))
    if AUGMENTATION_SWITCHES.get('use_zoom_and_crop', False):
        transform_list.append(transforms.RandomResizedCrop(size=(224, 224), scale=config['zoom']))
    if AUGMENTATION_SWITCHES.get('use_perspective', False):
        transform_list.append(transforms.RandomPerspective(distortion_scale=config['perspective'], p=0.5))
    
    transform_list.append(transforms.ToTensor())
    if AUGMENTATION_SWITCHES.get('use_gaussian_noise', False):
        transform_list.append(AddGaussianNoise(mean=0., std=config['noise_std']))
    
    transform_list.append(transforms.ToPILImage())
    
    return transforms.Compose(transform_list)

# --- Testing and Visualization Functions ---
def test_single_transformation(image_path, output_dir, intensity_controls):
    """
    Applies and saves each transformation individually to an image for inspection.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path).convert("L")


    # Get a dummy class name for default augmentations
    class_name = 'No Finding'
    
    # Get individual transforms
    transforms_dict = {
        'original': transforms.Compose([GrayscaleTo3Channel()]),
        'histogram_equalization': transforms.Compose([HistogramEqualization(), GrayscaleTo3Channel()]),
        'rotation': transforms.Compose([GrayscaleTo3Channel(), transforms.RandomRotation(intensity_controls.get('rotation', 10))]),
        'zoom_crop': transforms.Compose([GrayscaleTo3Channel(), transforms.RandomResizedCrop(size=(224, 224), scale=intensity_controls.get('zoom', (0.9, 1.1)))]),
        'perspective': transforms.Compose([GrayscaleTo3Channel(), transforms.RandomPerspective(distortion_scale=intensity_controls.get('perspective', 0.1), p=1.0)]),
        'gaussian_noise': transforms.Compose([GrayscaleTo3Channel(), transforms.ToTensor(), AddGaussianNoise(mean=0., std=intensity_controls.get('noise_std', 0.05)), transforms.ToPILImage()]),

    }

    for name, trans in transforms_dict.items():
        # Ensure image is in the correct format for each transform
        img_for_transform = image.copy()
        augmented_image = trans(img_for_transform)
        save_path = os.path.join(output_dir, f"{base_name}_{name}.png")
        augmented_image.save(save_path)
    print(f"Saved individual transformations for {image_path} in {output_dir}")

def save_augmented_samples(image_path, output_dir, class_name, intensity_controls, num_samples=5):
    """
    Saves a number of augmented samples for a given image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path)
    augmentation = get_augmentations(class_name, intensity_controls)
    
    for i in range(num_samples):
        augmented_image = augmentation(image)
        save_path = os.path.join(output_dir, f"{base_name}_augmented_sample_{i}.png")
        augmented_image.save(save_path)
    print(f"Saved {num_samples} augmented samples for {image_path} in {output_dir}")

def visualize_and_adjust(image_path, class_name, intensity_controls):
    """
    Applies augmentations with given intensities and returns the augmented image.
    Useful for interactive environments like Jupyter notebooks.
    """
    image = Image.open(image_path)
    augmentation = get_augmentations(class_name, intensity_controls)
    augmented_image = augmentation(image)
    return augmented_image

# --- TensorBoard Integration ---
def log_augmentations_to_tensorboard(writer, tag, image_path, class_name, intensity_controls, num_samples=4):
    """
    Logs a grid of augmented images to TensorBoard.
    """
    image = Image.open(image_path)
    augmentation = get_augmentations(class_name, intensity_controls)
    
    # Create a grid of images
    grid = []
    for _ in range(num_samples):
        augmented_image = augmentation(image)
        # Convert PIL image to tensor
        grid.append(transforms.ToTensor()(augmented_image))
        
    image_grid = torchvision.utils.make_grid(grid)
    writer.add_image(tag, image_grid)

# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    # This is where a user can easily adjust the intensity of transformations.
    # These values will override the class-specific defaults.
    # To use defaults, keep this dictionary empty.
    custom_intensity_controls = {
        # 'rotation': 25,
        # 'zoom': (0.7, 1.3),
        # 'perspective': 0.25,
        # 'noise_std': 0.1,

    }

    # --- 1. Test Individual Transformations ---
    # This helps to understand the effect of each augmentation.
    print("--- Testing Individual Transformations ---")
    sample_images_dir = 'Filtered_CXR_Dataset' # Using a directory with sample images
    output_test_dir = 'augmentation_tests'
    if os.path.exists(output_test_dir):
        shutil.rmtree(output_test_dir)
    
    # Find one sample image per class
    sample_images = {
        'Mass': os.path.join(sample_images_dir, 'Mass', '00000012_000.png'),
        'Nodule': os.path.join(sample_images_dir, 'Nodule', '00000013_022.png'),
        'No Finding': os.path.join(sample_images_dir, 'No Finding', '00000013_023.png')
    }

    for class_name, image_path in sample_images.items():
        if os.path.exists(image_path):
            class_output_dir = os.path.join(output_test_dir, class_name)
            test_single_transformation(image_path, class_output_dir, custom_intensity_controls)
        else:
            print(f"Sample image not found for class {class_name} at {image_path}")

    # --- 2. Save Augmented Samples ---
    # This shows the result of the full augmentation pipeline.
    print("\n--- Saving Augmented Samples ---")
    output_samples_dir = 'augmented_samples'
    if os.path.exists(output_samples_dir):
        shutil.rmtree(output_samples_dir)

    for class_name, image_path in sample_images.items():
        if os.path.exists(image_path):
            class_output_dir = os.path.join(output_samples_dir, class_name)
            save_augmented_samples(image_path, class_output_dir, class_name, custom_intensity_controls)
        else:
            print(f"Sample image not found for class {class_name} at {image_path}")

    # --- 3. TensorBoard Logging Example ---
    # This demonstrates how to log augmented images for real-time monitoring.
    print("\n--- Logging to TensorBoard ---")
    writer = SummaryWriter('runs/augmentation_demo')
    
    for class_name, image_path in sample_images.items():
        if os.path.exists(image_path):
            log_augmentations_to_tensorboard(writer, f'Augmentations/{class_name}', image_path, class_name, custom_intensity_controls)
        else:
            print(f"Sample image not found for class {class_name} at {image_path}")
            
    writer.close()
    print("Logged augmented image samples to TensorBoard. To view, run: tensorboard --logdir=runs")

    # --- Interactive Visualization Example ---
    # The visualize_and_adjust function can be used in a Jupyter Notebook like this:
    #
    # from dataOgmentation import visualize_and_adjust
    # from PIL import Image
    # 
    # my_image_path = 'path/to/your/image.png'
    # my_class = 'Mass'
    # my_intensities = {'rotation': 30, 'noise_std': 0.15}
    # 
    # augmented_img = visualize_and_adjust(my_image_path, my_class, my_intensities)
    # augmented_img.show() # Display the image
    print("\n--- Interactive Visualization ---")
    print("The 'visualize_and_adjust' function is available for use in interactive environments like Jupyter notebooks.")
