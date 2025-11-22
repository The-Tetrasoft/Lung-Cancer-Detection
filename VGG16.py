import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torchvision

# --- Local Imports ---
from dataOgmentation import get_augmentations, log_augmentations_to_tensorboard

# --- CONFIGURATION ---
DATA_ROOT = 'D:/Task/CRX_Model_Dataset'
MODELS_OUTPUT_DIR = 'D:/Task/Trained_Models'
PREDICTIONS_OUTPUT_DIR = 'D:/Task/Model_Predictions'
MODEL_NAME = 'vgg16'

NUM_CLASSES = 3
BATCH_SIZE = 4
NUM_EPOCHS = 45
LEARNING_RATE = 0.001

# --- Fine-tuning and Scheduler Configuration ---
FINETUNE_FREEZE_EPOCHS = 5
LR_SCHEDULER_STEP_SIZE = 7
EARLY_STOPPING_PATIENCE = 10
LABEL_SMOOTHING_FACTOR = 0.01

# --- Augmentation Configuration ---
AUGMENTATION_SAMPLES_DIR = 'D:/Task/Augmentation_Samples/vgg16'

# --- SCRIPT SETUP ---

def clear_memory():
    """
    Clear GPU and CPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

class GrayscaleTo3Channel(object):
    """Converts a grayscale PIL image to a 3-channel PIL image."""
    def __call__(self, img):
        return img.convert('RGB')

class HistogramEqualization(object):
    """Applies histogram equalization to a grayscale PIL image."""
    def __call__(self, img):
        return ImageOps.equalize(img.convert('L'))

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    plt.close(fig)
    return fig

# --- Custom Dataset for Class-Specific Augmentations ---
class CustomAugmentationDataset(Dataset):
    """
    A custom dataset that applies class-specific augmentations.
    """
    def __init__(self, subset, class_names, intensity_controls, is_train=True):
        self.subset = subset
        self.is_train = is_train
        self.class_names = class_names
        self.intensity_controls = intensity_controls
        
        # Basic transforms for validation/test
        self.test_transform = transforms.Compose([
            HistogramEqualization(),
            GrayscaleTo3Channel(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_path, label_idx = self.subset.dataset.samples[self.subset.indices[index]]
        image = Image.open(img_path)
        
        if self.is_train:
            class_name = self.class_names[label_idx]
            augmentation = get_augmentations(class_name, self.intensity_controls)
            # The base augmentation pipeline from get_augmentations ends with ToPILImage()
            # We need to convert it back to a tensor and normalize
            final_transform = transforms.Compose([
                augmentation,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = final_transform(image)
        else:
            image = self.test_transform(image)
            
        return image, label_idx

    def __len__(self):
        return len(self.subset)

def get_model(model_name, num_classes, pretrained=True):
    """
    Loads a pretrained model from torchvision and replaces its final layer.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    print(f"✅ Loaded {model_name} with pretrained weights.")
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, patience, writer, class_names):
    """
    The main training loop. It now includes a validation phase after each epoch
    and saves the best performing model based on validation accuracy.
    """
    model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        clear_memory() # Clear memory at the start of each epoch
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset))

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Loss/val', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/val', val_epoch_acc, epoch)

        # --- Per-class accuracy and confusion matrix for validation set ---
        all_labels = []
        all_preds = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # Log per-class accuracy
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        for class_name in class_names:
            if class_name in report:
                writer.add_scalar(f'Accuracy/val_{class_name}', report[class_name]['recall'], epoch)

        # Log confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        fig = plot_confusion_matrix(cm, class_names)
        writer.add_figure('Confusion Matrix/validation', fig, epoch)

        # --- Visualize validation set predictions ---
        # Get a batch of validation images
        try:
            inputs, labels = next(iter(val_loader))
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Make a grid of images
            img_grid = torchvision.utils.make_grid(inputs.cpu())
            writer.add_image('Validation/images', img_grid, epoch)
            
            # Add labels to the image
            text_summary = "True: " + ", ".join([class_names[i] for i in labels]) + "\n\nPred: " + ", ".join([class_names[i] for i in preds])
            writer.add_text('Validation/labels', text_summary, epoch)
        except StopIteration:
            print("Could not fetch a batch from validation loader for visualization.")

        print(f"  -> Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f}")

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  -> 🎉 New best model found with validation accuracy: {best_acc:.4f}!")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s). Patience: {patience}.")
        
        scheduler.step()

        if epochs_no_improve >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs with no improvement.")
            break
    
    model.load_state_dict(best_model_wts)
    return model

def evaluate_and_save_predictions(model, test_loader, device, class_names, model_name):
    """
    Evaluates the model on the test set, prints a classification report,
    and saves the predictions to a CSV file.
    """
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_filenames = []

    progress_bar = tqdm(test_loader, desc="[Evaluating]")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            start_index = i * test_loader.batch_size
            end_index = start_index + len(labels)
            batch_filenames = [os.path.basename(path) for path, _ in test_loader.dataset.samples[start_index:end_index]]
            all_filenames.extend(batch_filenames)

    print("\n--- Evaluation Report ---")
    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    print(report_str)

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    report_path = os.path.join(MODELS_OUTPUT_DIR, f'{model_name}_classification_report.json')
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"✅ Classification report saved to {report_path}")

    prob_columns = [f'prob_{name}' for name in class_names]
    df_preds = pd.DataFrame(all_probs, columns=prob_columns)
    df_preds['filename'] = all_filenames
    df_preds['true_label'] = [class_names[i] for i in all_labels]
    df_preds['predicted_label'] = [class_names[i] for i in all_preds]
    
    df_preds = df_preds[['filename', 'true_label', 'predicted_label'] + prob_columns]
    
    pred_csv_path = os.path.join(PREDICTIONS_OUTPUT_DIR, f'{model_name}_predictions.csv')
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"✅ Predictions for diversity analysis saved to {pred_csv_path}")

def main(model_name):
    """
    Main function to orchestrate data loading, training, and evaluation.
    """
    print(f"--- Starting process for model: {model_name} ---")

    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)

    writer = SummaryWriter(f'runs/{model_name}')

    # --- Data Loading and Augmentation ---
    # Note: We load the dataset without default transforms, as our custom dataset will handle them.
    full_train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'Training_set'))
    test_dataset_folder = datasets.ImageFolder(os.path.join(DATA_ROOT, 'Testing_set'))
    class_names = full_train_dataset.classes
    print(f"✅ Datasets loaded. Classes: {class_names}")

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    # --- Intensity controls for augmentations ---
    # You can customize these values
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

    train_dataset = CustomAugmentationDataset(train_subset, class_names, intensity_controls, is_train=True)
    val_dataset = CustomAugmentationDataset(val_subset, class_names, intensity_controls, is_train=False)
    
    # For test dataset, we create a subset-like object to use with our custom dataset class
    test_indices = list(range(len(test_dataset_folder)))
    test_subset = torch.utils.data.Subset(test_dataset_folder, test_indices)
    test_dataset = CustomAugmentationDataset(test_subset, class_names, intensity_controls, is_train=False)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Log sample augmented images to TensorBoard ---
    print("Logging sample augmented images to TensorBoard...")
    sample_images_dir = 'd:/Task/Filtered_CXR_Dataset'
    sample_images = {
        'Mass': os.path.join(sample_images_dir, 'Mass', '00000012_000.png'),
        'Nodule': os.path.join(sample_images_dir, 'Nodule', '00000013_022.png'),
        'No Finding': os.path.join(sample_images_dir, 'No Finding', '00000013_023.png')
    }
    for class_name, image_path in sample_images.items():
        if os.path.exists(image_path):
            log_augmentations_to_tensorboard(writer, f'Augmentations/{class_name}', image_path, class_name, intensity_controls)

    class_names = full_train_dataset.classes
    print(f"✅ Datasets loaded. Classes: {class_names}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(model_name, NUM_CLASSES)

    class_counts = [0] * len(class_names)
    for _, label_idx in full_train_dataset.samples:
        class_counts[label_idx] += 1
    
    total_samples = float(sum(class_counts))
    class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Using class weights for loss function: {class_weights_tensor.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING_FACTOR)

    print("\n--- Stage 1: Training only the final layer ---")
    clear_memory()
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_warmup = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler_warmup = optim.lr_scheduler.StepLR(optimizer_warmup, step_size=LR_SCHEDULER_STEP_SIZE, gamma=0.1)
    model = train_model(model, train_loader, val_loader, criterion, optimizer_warmup, scheduler_warmup, device, FINETUNE_FREEZE_EPOCHS, patience=EARLY_STOPPING_PATIENCE, writer=writer, class_names=class_names)

    print("\n--- Stage 2: Fine-tuning the entire model ---")
    clear_memory()
    for param in model.parameters():
        param.requires_grad = True
    optimizer_finetune = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)
    scheduler_finetune = optim.lr_scheduler.StepLR(optimizer_finetune, step_size=LR_SCHEDULER_STEP_SIZE, gamma=0.1)
    model = train_model(model, train_loader, val_loader, criterion, optimizer_finetune, scheduler_finetune, device, NUM_EPOCHS - FINETUNE_FREEZE_EPOCHS, patience=EARLY_STOPPING_PATIENCE, writer=writer, class_names=class_names)

    model_save_path = os.path.join(MODELS_OUTPUT_DIR, f'{model_name}_best.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Trained model weights saved to: {model_save_path}")

    evaluate_and_save_predictions(model, test_loader, device, class_names, model_name)

    print(f"\n--- Process for {model_name} finished successfully! ---")
    writer.close()


if __name__ == '__main__':
    main(MODEL_NAME)
