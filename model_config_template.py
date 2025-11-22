# Configuration template for all models
MODEL_CONFIG_TEMPLATE = {
    'batch_size': 4,  # Reduced to prevent memory issues
    'num_workers': 2,  # Reduced worker threads
    'intensity_controls': {
        'Mass': {
            'rotation': 15,
            'zoom': (0.85, 1.15),
            'perspective': 0.15,
            'noise_std': 0.1,
            'erase_scale': (0.02, 0.1),
            'elastic_alpha': 50,
            'elastic_sigma': 5
        },
        'Nodule': {
            'rotation': 5,
            'zoom': (0.95, 1.05),
            'perspective': 0.05,
            'noise_std': 0.05,
            'erase_scale': (0.01, 0.03),
            'elastic_alpha': 25,
            'elastic_sigma': 3
        },
        'No Finding': {
            'rotation': 20,
            'zoom': (0.8, 1.2),
            'perspective': 0.2,
            'noise_std': 0.15,
            'erase_scale': (0.02, 0.15),
            'elastic_alpha': 60,
            'elastic_sigma': 6
        }
    }
}

def apply_model_fixes(model_file, model_name):
    """
    Updates a model file with memory management, proper augmentation controls,
    and consistent configuration.
    """
    updates = [
        {
            'section': 'imports',
            'add': [
                'import gc',
                'from torch.cuda.amp import GradScaler, autocast',
                'from dataOgmentation import get_augmentations, log_augmentations_to_tensorboard, save_augmented_samples'
            ]
        },
        {
            'section': 'configuration',
            'vars': {
                'BATCH_SIZE': 4,
                'NUM_WORKERS': 2,
                'CLEAR_MEMORY': True,
                'AUGMENTATION_SAMPLES_DIR': f'./Augmentation_Samples/{model_name}'
            }
        },
        {
            'section': 'functions',
            'add': [
                'def clear_memory():',
                '    """Clear GPU and CPU memory."""',
                '    if torch.cuda.is_available():',
                '        torch.cuda.empty_cache()',
                '    gc.collect()',
            ]
        }
    ]
    return updates