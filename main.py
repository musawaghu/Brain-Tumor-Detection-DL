import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import re
import csv
import random
from tqdm import tqdm


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# Define constants
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
# Relative paths so the project runs on any machine (put the PMRAM images in ./data/)
IMAGE_DIR = "data"
CSV_FILE = "brain_tumor_labels.csv"

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class BrainMRIDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        """
        Args:
            img_dir: Directory with all the images
            csv_file: Path to the CSV file with image filenames and numerical labels
            transform: Optional transform to be applied on images
        """
        self.img_dir = img_dir
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform

        # Get unique labels (already numerical)
        self.unique_labels = sorted(self.labels_df.iloc[:, 1].unique())
        self.num_classes = len(self.unique_labels)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]  # Assuming first column is filename
        img_path = os.path.join(self.img_dir, img_name)

        # Handle cases where image might be in a subfolder
        if not os.path.exists(img_path):
            # Try to find the image recursively
            for root, _, files in os.walk(self.img_dir):
                if img_name in files:
                    img_path = os.path.join(root, img_name)
                    break

        image = Image.open(img_path).convert('RGB')

        # Get the numerical label directly
        label = int(self.labels_df.iloc[idx, 1])  # Convert to int to ensure it's an integer

        if self.transform:
            image = self.transform(image)

        return image, label


# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Stratified 70/15/15 split. The test set is never seen during training or model selection,
# so its score is an honest estimate of performance on unseen images.
def split_data(df, seed=42):
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed, stratify=df.iloc[:, 1])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df.iloc[:, 1])
    return train_df, val_df, test_df


# Function to prepare data
def prepare_data():
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    # Read CSV and display basic info
    df = pd.read_csv(CSV_FILE)
    print(f"Dataset size: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")

    # Split into train / validation / test sets (70 / 15 / 15)
    train_df, val_df, test_df = split_data(df)

    # Save the splits for reference
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    test_df.to_csv('test_split.csv', index=False)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Create temporary CSV files for the train, validation and test datasets
    train_csv = 'train_temp.csv'
    val_csv = 'val_temp.csv'
    test_csv = 'test_temp.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Create the datasets
    train_dataset = BrainMRIDataset(img_dir=IMAGE_DIR, csv_file=train_csv, transform=train_transform)
    val_dataset = BrainMRIDataset(img_dir=IMAGE_DIR, csv_file=val_csv, transform=val_transform)
    test_dataset = BrainMRIDataset(img_dir=IMAGE_DIR, csv_file=test_csv, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # Speeds up the CPU to GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Clean up temporary files
    os.remove(train_csv)
    os.remove(val_csv)
    os.remove(test_csv)

    # Get numerical class labels
    unique_labels = sorted(df.iloc[:, 1].unique())
    num_classes = len(unique_labels)

    # Use numeric values directly as class names
    class_names = [str(label) for label in unique_labels]

    print(f"Number of classes: {num_classes}")
    print(f"Unique labels: {unique_labels}")

    return train_loader, val_loader, test_loader, class_names, num_classes


# Get a pretrained ResNet model and modify for our task
def get_model(num_classes):
    # Load a pretrained ResNet50 model
    model = models.resnet50(weights='IMAGENET1K_V2')

    # Freeze early layers to avoid overfitting
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes)
    )

    return model.to(DEVICE)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_acc = 0.0
    best_val_recall = 0.0  # Added to track best recall
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_recall': []  # Added recall tracking
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for inputs, labels in train_pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.cpu().numpy())

        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []

        # No gradients needed for validation
        with torch.no_grad():
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")

            for inputs, labels in val_pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Save predictions and labels for metrics calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar
                val_pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        # Calculate recall score
        epoch_recall = recall_score(all_labels, all_preds, average='macro')

        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.cpu().numpy())
        history['val_recall'].append(epoch_recall)

        print(f"Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}, Val Recall: {epoch_recall:.4f}")

        # Save best model based on accuracy
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_val_acc,
                'recall': epoch_recall,
            }, 'best_acc_model.pth')
            print(f"Saved new best accuracy model: {best_val_acc:.4f}")

        # Also save best model based on recall
        if epoch_recall > best_val_recall:
            best_val_recall = epoch_recall
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': epoch_acc,
                'recall': best_val_recall,
            }, 'best_recall_model.pth')
            print(f"Saved new best recall model: {best_val_recall:.4f}")

        # Update learning rate
        scheduler.step(epoch_loss)

    return model, history


# Function for visualizing training results
def plot_training_history(history):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    # Plot recall
    ax3.plot(history['val_recall'], label='Validation Recall', color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Recall')
    ax3.set_title('Validation Recall')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))

    # Create a prettier visualization with seaborn
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Calculate per-class metrics
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        class_recall = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        class_precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (
                                                                                                          class_precision + class_recall) > 0 else 0

        print(f"Class {class_name}:")
        print(f"  Recall: {class_recall:.4f}")
        print(f"  Precision: {class_precision:.4f}")
        print(f"  F1-score: {class_f1:.4f}")
        print(f"  Support: {np.sum(cm[i, :])}")

    return cm


# Function to make predictions on new images
def predict_image(model, image_path, class_names, transform=val_transform):
    model.eval()

    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Get the predicted class and probability
    pred_idx = predicted.item()
    pred_class = class_names[pred_idx]
    pred_prob = probabilities[pred_idx].item()

    # Display the image with the prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Predicted: {pred_class} ({pred_prob:.2%})")
    plt.axis('off')
    plt.show()

    # Print all class probabilities
    print("Class Probabilities:")
    for i, (cls, prob) in enumerate(zip(class_names, probabilities)):
        print(f"{cls}: {prob:.2%}")

    return pred_class, pred_prob


# Bootstrap 95% confidence intervals: resample the evaluated images with replacement many times,
# recompute each metric, and take the 2.5th and 97.5th percentiles. Shows how much the score
# could move with a different sample of test images.
def bootstrap_ci(y_true, y_pred, y_score, n_boot=1000, seed=42):
    y_true, y_pred, y_score = np.array(y_true), np.array(y_pred), np.array(y_score)
    rng = np.random.default_rng(seed)
    metric_fns = {
        'accuracy': lambda t, p, s: accuracy_score(t, p),
        'precision': lambda t, p, s: precision_score(t, p, average='macro', zero_division=0),
        'recall': lambda t, p, s: recall_score(t, p, average='macro', zero_division=0),
        'f1': lambda t, p, s: f1_score(t, p, average='macro', zero_division=0),
        'roc_auc': lambda t, p, s: roc_auc_score(t, s),
    }
    samples = {name: [] for name in metric_fns}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:  # ROC-AUC needs both classes in the resample
            continue
        for name, fn in metric_fns.items():
            samples[name].append(fn(y_true[idx], y_pred[idx], y_score[idx]))
    return {name: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for name, v in samples.items()}


# Function to evaluate model on a held-out set (validation or test)
def evaluate_model(model, data_loader, criterion, class_names):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    all_probs = []  # For ROC-AUC calculation

    # No gradients needed for evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Get probabilities for ROC-AUC
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Save predictions and labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate overall metrics
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Print overall metrics
    print(f"Evaluation Loss: {epoch_loss:.4f}")
    print(f"Accuracy: {epoch_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Convert all_labels to one-hot encoding for ROC-AUC if multi-class
    num_classes = len(class_names)
    if num_classes > 2:
        # Try to calculate multi-class ROC-AUC
        try:
            roc_auc = roc_auc_score(
                np.eye(num_classes)[all_labels],
                all_probs,
                multi_class='ovr',
                average='macro'
            )
            print(f"ROC-AUC Score (OvR): {roc_auc:.4f}")
        except Exception as e:
            print(f"Couldn't calculate ROC-AUC: {e}")
    else:
        # Binary classification
        try:
            roc_auc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        except Exception as e:
            print(f"Couldn't calculate ROC-AUC: {e}")

    # 95% confidence intervals (binary task: score = probability of the tumor class)
    cis = bootstrap_ci(all_labels, all_preds, [prob[1] for prob in all_probs])
    print("95% confidence intervals (bootstrap, 1000 resamples):")
    for name, (low, high) in cis.items():
        print(f"  {name}: [{low:.4f}, {high:.4f}]")

    # Plot confusion matrix
    cm = plot_confusion_matrix(all_labels, all_preds, class_names)

    return epoch_loss, epoch_acc.item(), recall, all_preds, all_labels, cm, cis


def main():
    print("Preparing data...")
    train_loader, val_loader, test_loader, class_names, num_classes = prepare_data()

    print("Initializing model...")
    model = get_model(num_classes)

    # Print model summary
    print(f"Model architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LEARNING_RATE, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    print("Starting training...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
    )

    # Plot training history
    plot_training_history(history)

    # Use the epoch that scored best on validation (not simply the last epoch)
    # weights_only=False: our own file, and it also stores optimizer state and metric values
    best_checkpoint = torch.load('best_acc_model.pth', map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"Loaded best validation model from epoch {best_checkpoint['epoch'] + 1}")

    # Evaluate exactly once on the held-out test set
    print("Evaluating best model on held-out test set...")
    test_loss, test_acc, test_recall, all_preds, all_labels, cm, test_cis = evaluate_model(
        model, test_loader, criterion, class_names
    )
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'test_accuracy': test_acc,
        'test_recall': float(test_recall),
        'test_cis': test_cis,
        'confusion_matrix': cm.tolist(),
    }, 'final_brain_mri_model.pth')

    print("Training complete!")
    print("Models have been saved as:")
    print("- 'best_acc_model.pth' (best accuracy)")
    print("- 'best_recall_model.pth' (best recall)")
    print("- 'final_brain_mri_model.pth' (best validation model + test metrics, used by app.py)")

    # Example of how to use the model for prediction (uncomment when needed)
    # example_image = "path/to/example/image.jpg"
    # predicted_class, probability = predict_image(model, example_image, class_names)
    # print(f"Predicted class: {predicted_class} with probability {probability:.2%}")


# Function to load and use the trained model
def load_and_predict(model_path, image_path):
    # Load the saved model
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Initialize model with the right number of classes
    num_classes = len(checkpoint['class_names'])
    model = get_model(num_classes)

    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Predict using the loaded model
    predicted_class, probability = predict_image(
        model, image_path, checkpoint['class_names']
    )

    return predicted_class, probability


if __name__ == "__main__":
    main()