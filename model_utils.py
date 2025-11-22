import torch
import os
from torch.utils.tensorboard import SummaryWriter
import gc

# Define constants that were missing.
# These might be better in a separate config file.
DATA_ROOT = "CRX_Model_Dataset"
MODELS_OUTPUT_DIR = "Trained_Models"
PREDICTIONS_OUTPUT_DIR = "Model_Predictions"
AUGMENTATION_SAMPLES_DIR = "Augmentation_Samples"
CLEAR_MEMORY = True

def clear_memory():
    """
    Clear GPU and CPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def check_paths():
    """
    Verify all required paths exist.
    """
    required_paths = [
        DATA_ROOT,
        os.path.join(DATA_ROOT, 'Training_set'),
        os.path.join(DATA_ROOT, 'Testing_set'),
        MODELS_OUTPUT_DIR,
        PREDICTIONS_OUTPUT_DIR
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path does not exist: {path}")

def main(model_name):
    """
main function to orchestrate data loading, training, and evaluation.
    """
    try:
        print(f"--- Starting process for model: {model_name} ---")
        
        # Check paths before starting
        check_paths()
        
        # Clear memory at start
        if CLEAR_MEMORY:
            clear_memory()

        # Create required directories
        os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
        os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)
        os.makedirs(AUGMENTATION_SAMPLES_DIR, exist_ok=True)

        # Initialize TensorBoard writer
        writer = SummaryWriter(f'runs/{model_name}')
    except Exception as e:
        print(f"An error occurred in main function: {e}")

if __name__ == '__main__':
    # This is an example of how you might run this main function.
    # You would typically call this from another script.
    main('default_model')