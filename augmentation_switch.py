# d:/Task/augmentation_switch.py

# This file contains global switches to enable or disable specific data augmentations
# across all training scripts that use the `dataOgmentation.py` module.
#
# To turn an augmentation OFF, simply set its value to False.

AUGMENTATION_SWITCHES = {
    'use_horizontal_flip': True,
    'use_rotation': True,
    'use_zoom_and_crop': False,
    'use_perspective': True,
    'use_gaussian_noise': False,
}