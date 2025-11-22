import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import torchvision
import random
from model_utils import clear_memory, check_paths

# --- CONFIGURATION ---
DATA_ROOT = 'D:/Task/CRX_Model_Dataset'
MODELS_OUTPUT_DIR = 'D:/Task/Trained_Models'
PREDICTIONS_OUTPUT_DIR = 'D:/Task/Model_Predictions'
AUGMENTATION_SAMPLES_DIR = 'D:/Task/Augmentation_Samples/densenet169'
MODEL_NAME = 'densenet169'

NUM_CLASSES = 3
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# --- Fine-tuning and Scheduler Configuration ---
FINETUNE_FREEZE_EPOCHS = 5
LR_SCHEDULER_STEP_SIZE = 7
EARLY_STOPPING_PATIENCE = 10
LABEL_SMOOTHING_FACTOR = 0.1

# --- AUGMENTATION INTENSITY CONTROLS ---
intensity_controls = {
    'Mass': {
        'random_perspective_distortion': 0.6,
        'random_rotation_degrees': 30,
        'random_horizontal_flip_p': 0.7,
        'random_vertical_flip_p': 0.7,
        'random_zoom_scale': (0.7, 1.4),
        'gaussian_noise_std': 0.03,
    },
    'Nodule': {
        'random_perspective_distortion': 0.4,
        'random_rotation_degrees': 15,
        'random_horizontal_flip_p': 0.5,
        'random_vertical_flip_p': 0.5,
        'random_zoom_scale': (0.8, 1.2),
        'gaussian_noise_std': 0.015,
    },
    'No Finding': {
        'random_perspective_distortion': 0.5,
        'random_rotation_degrees': 20,
        'random_horizontal_flip_p': 0.5,
        'random_vertical_flip_p': 0.5,
        'random_zoom_scale': (0.8, 1.3),
        'gaussian_noise_std': 0.02,
    }
}

# --- CUSTOM TRANSFORMS ---

class GrayscaleTo3Channel(object):
    def __call__(self, img):
        return img.convert('RGB')

class HistogramEqualization(object):
    def __call__(self, img):
        return ImageOps.equalize(img.convert('L'))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class RandomZoom(object):
    def __init__(self, scale=(0.8, 1.2)):
        self.scale = scale

    def __call__(self, img):
        scale = random.uniform(self.scale[0], self.scale[1])
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        return img.crop((x1, y1, x1 + w, y1 + h))

# --- AUGMENTATION PIPELINE ---

def get_augmentations(class_name, controls):
    class_controls = controls.get(class_name, controls['No Finding'])

    return transforms.Compose([
        HistogramEqualization(),
        GrayscaleTo3Channel(),
        transforms.Resize((256, 256)),
        transforms.RandomPerspective(distortion_scale=class_controls['random_perspective_distortion'], p=0.5),
        transforms.RandomRotation(class_controls['random_rotation_degrees']),
        transforms.RandomHorizontalFlip(p=class_controls['random_horizontal_flip_p']),
        transforms.RandomVerticalFlip(p=class_controls['random_vertical_flip_p']),
        RandomZoom(scale=class_controls['random_zoom_scale']),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        AddGaussianNoise(std=class_controls['gaussian_noise_std']),
        transforms.ToPILImage()
    ])

# --- DATASET ---

class CustomAugmentationDataset(Dataset):
    def __init__(self, subset, class_names, intensity_controls, is_train=True):
        self.subset = subset
        self.is_train = is_train
        self.class_names = class_names
        self.intensity_controls = intensity_controls
        
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

# --- MODEL ---

def get_model(model_name, num_classes, pretrained=True):
    model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    print(f"✅ Loaded {model_name} with pretrained weights.")
    return model

# --- UTILS ---

def plot_confusion_matrix(cm, class_names):
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

def test_augmentations(dataset, class_names, controls, num_samples=3):
    os.makedirs(AUGMENTATION_SAMPLES_DIR, exist_ok=True)
    print(f"--- Testing Augmentations & Saving Samples to {AUGMENTATION_SAMPLES_DIR} ---")

    for class_idx, class_name in enumerate(class_names):
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
        sample_indices = random.sample(class_indices, min(num_samples, len(class_indices)))

        for i, sample_idx in enumerate(sample_indices):
            img_path, _ = dataset.samples[sample_idx]
            original_image = Image.open(img_path).resize((224, 224))
            original_image.save(os.path.join(AUGMENTATION_SAMPLES_DIR, f'{class_name}_{i}_original.png'))

            augmentation_pipeline = get_augmentations(class_name, controls)
            augmented_image = augmentation_pipeline(Image.open(img_path))
            augmented_image.save(os.path.join(AUGMENTATION_SAMPLES_DIR, f'{class_name}_{i}_augmented.png'))

    print("✅ Augmentation samples saved.")

def log_validation_predictions(writer, epoch, val_loader, model, device, class_names, num_images=16):
    model.eval()
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    img_grid = torchvision.utils.make_grid(images[:num_images])
    
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
    plt.title('Validation Set Predictions (Predicted vs. True)')
    plt.axis('off')
    
    writer.add_figure('Validation/Predictions_vs_Actuals', fig, epoch)
    grid_labels = [f'P:{class_names[p]} T:{class_names[t]}' for p, t in zip(preds, labels)]
    writer.add_text('Validation/Prediction_Labels', ' | '.join(grid_labels), epoch)
    model.train()

# --- TRAINING & EVALUATION ---

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, patience, writer, class_names):
    model.to(device)
    scaler = GradScaler()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            progress_bar.set_postfix(loss=running_loss / ((progress_bar.n + 1) * train_loader.batch_size))

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        print(f"  -> Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        log_validation_predictions(writer, epoch, val_loader, model, device, class_names)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  -> 🎉 New best model found with validation accuracy: {best_acc:.4f}!")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")
        
        scheduler.step()

        if epochs_no_improve >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs with no improvement.")
            break
    
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="[Evaluating]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc, all_labels, all_preds

def main(model_name, intensity_controls):
    print(f"--- Starting process for model: {model_name} ---")
    writer = SummaryWriter(log_dir=f'runs/{model_name}_{pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")}')

    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)

    base_train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'Training_set'))
    test_dataset_raw = datasets.ImageFolder(os.path.join(DATA_ROOT, 'Testing_set'))
    class_names = base_train_dataset.classes

    test_augmentations(base_train_dataset, class_names, intensity_controls)

    train_size = int(0.9 * len(base_train_dataset))
    val_size = len(base_train_dataset) - train_size
    train_subset_indices, val_subset_indices = torch.utils.data.random_split(range(len(base_train_dataset)), [train_size, val_size])
    
    train_subset = torch.utils.data.Subset(base_train_dataset, train_subset_indices.indices)
    val_subset = torch.utils.data.Subset(base_train_dataset, val_subset_indices.indices)

    train_dataset = CustomAugmentationDataset(train_subset, class_names, intensity_controls, is_train=True)
    val_dataset = CustomAugmentationDataset(val_subset, class_names, intensity_controls, is_train=False)
    test_dataset = CustomAugmentationDataset(torch.utils.data.Subset(test_dataset_raw, range(len(test_dataset_raw))), class_names, intensity_controls, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"✅ Datasets loaded. Classes: {class_names}")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(model_name, NUM_CLASSES)

    class_counts = np.bincount([s[1] for s in base_train_dataset.samples])
    class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING_FACTOR)

    print("\n--- Stage 1: Training only the final layer ---")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_warmup = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler_warmup = optim.lr_scheduler.StepLR(optimizer_warmup, step_size=LR_SCHEDULER_STEP_SIZE, gamma=0.1)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer_warmup, scheduler_warmup, device, FINETUNE_FREEZE_EPOCHS, EARLY_STOPPING_PATIENCE, writer, class_names)

    print("\n--- Stage 2: Fine-tuning the entire model ---")
    for param in model.parameters():
        param.requires_grad = True
    optimizer_finetune = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)
    scheduler_finetune = optim.lr_scheduler.StepLR(optimizer_finetune, step_size=LR_SCHEDULER_STEP_SIZE, gamma=0.1)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer_finetune, scheduler_finetune, device, NUM_EPOCHS - FINETUNE_FREEZE_EPOCHS, EARLY_STOPPING_PATIENCE, writer, class_names)

    model_save_path = os.path.join(MODELS_OUTPUT_DIR, f'{model_name}_best.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Trained model weights saved to: {model_save_path}")

    print("\n--- Final Evaluation on Test Set ---")
    test_loss, test_acc, test_labels, test_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    report_dict = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(test_labels, test_preds, target_names=class_names)
    print(report_str)
    
    writer.add_text('Evaluation/Classification_Report', '<pre>' + report_str.replace(' ', '&nbsp;').replace('\n', '<br/>') + '</pre>', 0)
    cm = confusion_matrix(test_labels, test_preds)
    cm_fig = plot_confusion_matrix(cm, class_names)
    writer.add_figure('Evaluation/Confusion_Matrix', cm_fig, 0)

    report_path = os.path.join(MODELS_OUTPUT_DIR, f'{model_name}_classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"✅ Classification report saved to {report_path}")

    writer.close()
    print(f"\n--- Process for {model_name} finished successfully! ---")
    print(f"📊 TensorBoard logs saved to runs/{writer.log_dir}")


if __name__ == '__main__':
    main(MODEL_NAME, intensity_controls)