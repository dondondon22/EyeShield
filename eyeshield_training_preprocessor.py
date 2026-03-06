"""
EyeShield: Sprint 3 - DR Classification Model Training
Diabetic Retinopathy Screening System with EDL-Based Uncertainty Rejection

Using Your Custom Image Processor (No CLAHE)
Training Script for Google Colab
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Framework
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torchmetrics
from torch.cuda.amp import autocast, GradScaler

# For data handling
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image
import pydicom

# GPU/Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==================== IMAGE PREPROCESSING (YOUR CODE) ====================
class ImagePreprocessor:
    """Image preprocessing for fundus images - Based on your image_processor.py"""
    
    def __init__(self, target_size=(512, 512)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
    
    def preprocess_fundus_image(self, image_path):
        """
        Preprocesses a fundus image by resizing it to the target size.
        Supports DICOM (.dcm) and standard image formats (jpg, png, etc.)
        """
        # Check file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        
        if file_ext == '.dcm':
            # Load DICOM file
            try:
                dicom = pydicom.dcmread(image_path)
                img = dicom.pixel_array
                
                # Convert to 8-bit if needed
                if img.dtype != np.uint8:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Convert grayscale to BGR for consistency with color images
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            except Exception as e:
                raise ValueError(f"Error reading DICOM file: {str(e)}")
        else:
            # Load standard image formats
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Image not found or invalid path: {image_path}")
        
        # Resize with high-quality interpolation
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0,1] for model input
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    
    def assess_image_quality(self, preprocessed_img, blur_threshold=70, 
                            brightness_low=30, brightness_high=220, entropy_high=7.5):
        """
        Assesses the quality of a preprocessed fundus image using heuristic criteria.
        Returns quality metrics and assessment result
        """
        img_uint8 = (preprocessed_img * 255).astype(np.uint8)
        
        # Handle grayscale or BGR
        if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_uint8 if len(img_uint8.shape) == 2 else cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Calculate metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = np.mean(gray)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        # Store metrics
        quality_info = {
            'blur_var': float(laplacian_var),
            'brightness': float(mean_brightness),
            'entropy': float(entropy),
            'blur_threshold': blur_threshold,
            'brightness_low': brightness_low,
            'brightness_high': brightness_high,
            'entropy_threshold': entropy_high
        }
        
        # Assess quality
        if laplacian_var < blur_threshold:
            quality_result = "Rejected: Blurry or out of focus"
            quality_score = 0.3
        elif mean_brightness < brightness_low:
            quality_result = "Rejected: Too dark"
            quality_score = 0.2
        elif mean_brightness > brightness_high:
            quality_result = "Rejected: Too bright"
            quality_score = 0.2
        elif entropy > entropy_high:
            quality_result = "Rejected: Artifacts or obstructions"
            quality_score = 0.4
        else:
            quality_result = "Gradable"
            quality_score = 0.9  # Good quality
        
        return quality_score, quality_result, quality_info
    
    def preprocess(self, image_path, assess_quality=True):
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to image file
            assess_quality: Whether to assess quality
            
        Returns:
            preprocessed_image: Numpy array (H, W, 3) normalized to [0, 1]
            quality_score: Image quality metric (0-1)
            quality_info: Detailed quality metrics
        """
        try:
            # Preprocess image
            image = self.preprocess_fundus_image(image_path)
            
            # Assess quality if requested
            if assess_quality:
                quality_score, quality_result, quality_info = self.assess_image_quality(image)
            else:
                quality_score = 1.0
                quality_result = "Not assessed"
                quality_info = {}
            
            return image, quality_score, quality_info
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None, 0.0, {}
    
    def batch_preprocess(self, image_paths, verbose=True):
        """
        Preprocess multiple images
        
        Args:
            image_paths: List of image paths
            verbose: Print progress
            
        Returns:
            preprocessed_images: List of preprocessed images
            quality_scores: List of quality scores
            quality_info_list: List of quality info dicts
        """
        preprocessed = []
        quality_scores = []
        quality_info_list = []
        
        iterator = tqdm(image_paths) if verbose else image_paths
        
        for img_path in iterator:
            img, quality, info = self.preprocess(img_path, assess_quality=True)
            
            if img is not None:
                preprocessed.append(img)
                quality_scores.append(quality)
                quality_info_list.append(info)
        
        return preprocessed, quality_scores, quality_info_list


# ==================== CONFIGURATION ====================
class Config:
    """Configuration for training"""
    
    # Image Preprocessing
    TARGET_IMAGE_SIZE = (512, 512)  # Your preferred size for training
    QUALITY_CHECK = True            # Assess image quality
    
    # Dataset paths
    KAGGLE_DATASET_PATH = '/kaggle/input'
    COLAB_DRIVE_PATH = '/content/drive/MyDrive'
    
    # Model parameters
    NUM_CLASSES = 5  # Grade 0-4: No DR, Mild, Moderate, Severe, Proliferative
    INPUT_SIZE = (512, 512)
    BATCH_SIZE = 96
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # EDL parameters
    EDL_UNCERTAINTY_THRESHOLD = 0.3
    KL_WEIGHT = 0.1
    ANNEALING_START = 10
    ANNEALING_STEP = 1.0
    
    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Checkpoint and logging
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    SAVE_INTERVAL = 5
    
    # Augmentation
    AUGMENT = True
    RANDOM_SEED = 42
    NUM_WORKERS = 4


# ==================== DATA LOADING WITH YOUR PREPROCESSOR ====================
class DiabeticRetinopathyDataset(Dataset):
    """Custom Dataset for Diabetic Retinopathy Images with your preprocessor"""
    
    def __init__(self, df, img_dir, transform=None, preprocessor=None):
        """
        Args:
            df: DataFrame with columns ['image_path', 'diagnosis']
            img_dir: Directory containing images
            transform: Optional transforms to apply
            preprocessor: ImagePreprocessor instance
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.preprocessor = preprocessor
        self.classes = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row['image_path'])
            
            # Check if file exists
            if not os.path.isfile(img_path):
                print(f"Warning: Image not found: {img_path}")
                # Return blank tensor if file missing
                blank = torch.zeros(3, *self.preprocessor.target_size)
                return blank, int(row['diagnosis'])
            
            img, quality_score, quality_info = self.preprocessor.preprocess(
                img_path, assess_quality=Config.QUALITY_CHECK
            )
            
            # Convert to PIL for transforms
            pil = Image.fromarray((img * 255).astype(np.uint8))
            
            if self.transform is not None:
                pil = self.transform(pil)
            
            label = int(row['diagnosis'])
            return pil, label
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            blank = torch.zeros(3, *self.preprocessor.target_size)
            return blank, 0


def get_data_transforms(augment=True):
    """Data augmentation and normalization"""
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            normalize
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


# ==================== EDL MODEL ====================
class EvidentialHead(nn.Module):
    """Evidential Deep Learning Head for Uncertainty Estimation"""
    
    def __init__(self, input_features, num_classes):
        super(EvidentialHead, self).__init__()
        self.num_classes = num_classes
        
        # Evidence layer with ReLU activation
        self.evidence_layer = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass returning evidence"""
        evidence = torch.relu(self.evidence_layer(x))
        return evidence


class EfficientNetB3EDL(nn.Module):
    """EfficientNet-B3 with EDL Head for DR Classification"""
    
    def __init__(self, num_classes=5, pretrained=True):
        super(EfficientNetB3EDL, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet-B3
        self.backbone = models.efficientnet_b3(pretrained=pretrained)
        
        # Get feature dimension
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with identity to get features
        self.backbone.classifier = nn.Identity()
        
        # EDL Head
        self.edl_head = EvidentialHead(num_features, num_classes)
    
    def forward(self, x):
        """Forward pass through backbone and EDL head"""
        features = self.backbone(x)
        evidence = self.edl_head(features)
        return evidence
    
    def predict(self, evidence):
        """Convert evidence to Dirichlet parameters and predictions"""
        alpha = evidence + 1  # α_k = e_k + 1
        S = torch.sum(alpha, dim=1, keepdim=True)  # Dirichlet strength
        belief = alpha / S  # Belief masses
        uncertainty = self.num_classes / S  # Total uncertainty
        confidence = 1 - uncertainty  # Confidence
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'belief': belief,
            'S': S,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'pred': torch.argmax(belief, dim=1)
        }


# ==================== LOSS FUNCTIONS ====================
class EvidentialLoss(nn.Module):
    """Type-II Maximum Likelihood EDL Loss"""
    
    def __init__(self, num_classes, kl_weight=0.1):
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
    
    def forward(self, evidence, targets, epoch, annealing_step):
        """Compute EDL loss with annealing KL regularization"""
        
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets_one_hot = torch.zeros(targets.size(0), self.num_classes, device=targets.device)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        else:
            targets_one_hot = targets
        
        # Dirichlet parameters
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Negative log-likelihood (MSE variant)
        belief = alpha / S
        nll_loss = torch.sum((targets_one_hot - belief) ** 2, dim=1, keepdim=True)
        nll_loss += torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        
        # KL regularization with annealing
        lam = min(1.0, epoch / annealing_step) if epoch < annealing_step else 1.0
        
        kl_alpha = (alpha - 1) * (1 - targets_one_hot)
        kl_loss = lam * torch.sum(
            torch.lgamma(torch.sum(alpha, dim=1, keepdim=True)) - 
            torch.sum(torch.lgamma(alpha), dim=1, keepdim=True) +
            torch.sum(kl_alpha * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=True),
            dim=1, keepdim=True
        )
        
        total_loss = torch.mean(nll_loss) + self.kl_weight * torch.mean(kl_loss)
        
        return total_loss, torch.mean(nll_loss), torch.mean(kl_loss)


# ==================== METRICS ====================
class EDLMetrics:
    """Metrics for EDL model evaluation"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.uncertainties = []
        self.confidences = []
    
    def update(self, output, targets):
        """Update metrics with batch results"""
        self.predictions.extend(output['pred'].cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.uncertainties.extend(output['uncertainty'].squeeze().cpu().detach().numpy())
        self.confidences.extend(output['confidence'].squeeze().cpu().detach().numpy())
    
    def compute(self):
        """Compute all metrics"""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        uncertainties = np.array(self.uncertainties)
        confidences = np.array(self.confidences)
        
        accuracy = accuracy_score(targets, preds)
        
        # Calibration error (Expected Calibration Error)
        conf_bins = np.linspace(0, 1, 11)
        ece = 0
        for i in range(len(conf_bins) - 1):
            mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
            if np.sum(mask) > 0:
                avg_conf = np.mean(confidences[mask])
                avg_acc = np.mean(preds[mask] == targets[mask])
                ece += np.abs(avg_conf - avg_acc) * np.sum(mask) / len(targets)
        
        return {
            'accuracy': accuracy,
            'ece': ece,
            'mean_uncertainty': np.mean(uncertainties),
            'mean_confidence': np.mean(confidences),
            'confusion_matrix': confusion_matrix(targets, preds, labels=range(self.num_classes))
        }


# ==================== TRAINING ====================
class Trainer:
    """Training loop for EDL model"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss and optimizer
        self.criterion = EvidentialLoss(config.NUM_CLASSES, kl_weight=config.KL_WEIGHT)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics
        self.metrics = EDLMetrics(config.NUM_CLASSES)
        
        # History
        self.history = {
            'train_loss': [], 'train_nll': [], 'train_kl': [], 'train_acc': [],
            'val_loss': [], 'val_nll': [], 'val_kl': [], 'val_acc': [], 'val_ece': []
        }
        
        # Create checkpoint directory
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0
        total_nll = 0
        total_kl = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                evidence = self.model(images)
                loss, nll_loss, kl_loss = self.criterion(
                    evidence, targets, epoch, self.config.ANNEALING_START
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Get predictions
            with torch.no_grad():
                output = self.model.predict(evidence)
                self.metrics.update(output, targets)
            
            total_loss += loss.item()
            total_nll += nll_loss.item()
            total_kl += kl_loss.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'nll': nll_loss.item(),
                'kl': kl_loss.item()
            })
        
        # Compute epoch metrics
        train_metrics = self.metrics.compute()
        
        return {
            'loss': total_loss / len(self.train_loader),
            'nll': total_nll / len(self.train_loader),
            'kl': total_kl / len(self.train_loader),
            'accuracy': train_metrics['accuracy']
        }
    
    def validate(self):
        """Validation loop"""
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0
        total_nll = 0
        total_kl = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                evidence = self.model(images)
                loss, nll_loss, kl_loss = self.criterion(
                    evidence, targets, self.config.NUM_EPOCHS, 
                    self.config.ANNEALING_START
                )
                
                output = self.model.predict(evidence)
                self.metrics.update(output, targets)
                
                total_loss += loss.item()
                total_nll += nll_loss.item()
                total_kl += kl_loss.item()
        
        val_metrics = self.metrics.compute()
        
        return {
            'loss': total_loss / len(self.val_loader),
            'nll': total_nll / len(self.val_loader),
            'kl': total_kl / len(self.val_loader),
            'accuracy': val_metrics['accuracy'],
            'ece': val_metrics['ece'],
            'confusion_matrix': val_metrics['confusion_matrix']
        }
    
    def train(self):
        """Full training loop"""
        best_val_acc = 0
        patience_counter = 0
        patience = 10
        
        print("\n" + "="*80)
        print("Starting Training: EfficientNet-B3 + EDL for DR Classification")
        print("Using Your Image Preprocessor")
        print("="*80 + "\n")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_nll'].append(train_metrics['nll'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_nll'].append(val_metrics['nll'])
            self.history['val_kl'].append(val_metrics['kl'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_ece'].append(val_metrics['ece'])
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | ECE: {val_metrics['ece']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch, val_metrics['accuracy'])
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                self.save_best_model(epoch, val_metrics)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
    
    def save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'accuracy': accuracy
        }
        path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'checkpoint_epoch_{epoch+1}_acc_{accuracy:.4f}.pt'
        )
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def save_best_model(self, epoch, val_metrics):
        """Save best model"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_metrics': val_metrics
        }
        path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pt')
        torch.save(checkpoint, path)
        print(f"Best model saved: {path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # NLL Loss
        axes[0, 1].plot(self.history['train_nll'], label='Train NLL')
        axes[0, 1].plot(self.history['val_nll'], label='Val NLL')
        axes[0, 1].set_title('NLL Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('NLL Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy
        axes[1, 0].plot(self.history['train_acc'], label='Train')
        axes[1, 0].plot(self.history['val_acc'], label='Val')
        axes[1, 0].set_title('Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # ECE (Calibration)
        axes[1, 1].plot(self.history['val_ece'], label='Calibration Error')
        axes[1, 1].set_title('Expected Calibration Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('ECE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.LOG_DIR, 'training_history.png'), dpi=300)
        print(f"Training history plot saved to {self.config.LOG_DIR}")
        plt.show()


# ==================== MAIN ====================
def main():
    """Main training function"""
    
    # Set random seed
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    print("\n" + "="*80)
    print("EyeShield: DR Classification Model Training (Sprint 3)")
    print("Using Your Image Preprocessor (No CLAHE)")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Num Classes: {Config.NUM_CLASSES}")
    print(f"  - Target Preprocessing Size: {Config.TARGET_IMAGE_SIZE}")
    print(f"  - Input Size: {Config.INPUT_SIZE}")
    print(f"  - Batch Size: {Config.BATCH_SIZE}")
    print(f"  - Num Epochs: {Config.NUM_EPOCHS}")
    print(f"  - Learning Rate: {Config.LEARNING_RATE}")
    print(f"  - EDL KL Weight: {Config.KL_WEIGHT}")
    print(f"  - Quality Check: {Config.QUALITY_CHECK}")
    print("="*80 + "\n")
    
    # Initialize preprocessor
    print("Initializing image preprocessor...")
    preprocessor = ImagePreprocessor(target_size=Config.TARGET_IMAGE_SIZE)
    print("✓ Image preprocessor initialized")
    print(f"  - Target size: {Config.TARGET_IMAGE_SIZE}")
    print(f"  - Quality assessment: {'Enabled' if Config.QUALITY_CHECK else 'Disabled'}\n")
    
    # Load dataset from CSV
    print("Loading dataset from CSV...")
    df = pd.read_csv('/content/dataset/labels.csv')
    dataset_root = '/tmp/kagglehub'  # Kaggle hub downloads to this directory
    
    # If using the Kaggle download from the notebook
    import kagglehub
    dataset_root = kagglehub.dataset_download("ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy")
    print(f"✓ Loaded {len(df)} images from dataset")
    print(f"  - Class distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")
    
    # Split data
    train_size = int(len(df) * Config.TRAIN_RATIO)
    val_size = int(len(df) * Config.VAL_RATIO)
    
    train_df = df[:train_size].reset_index(drop=True)
    val_df = df[train_size:train_size+val_size].reset_index(drop=True)
    test_df = df[train_size+val_size:].reset_index(drop=True)
    
    print(f"Data split:")
    print(f"  - Train: {len(train_df)} images")
    print(f"  - Val: {len(val_df)} images")
    print(f"  - Test: {len(test_df)} images\n")
    
    # Data transforms
    train_transform, val_transform = get_data_transforms(augment=Config.AUGMENT)
    
    # Datasets and dataloaders
    train_dataset = DiabeticRetinopathyDataset(
        train_df, dataset_root, transform=train_transform, preprocessor=preprocessor
    )
    val_dataset = DiabeticRetinopathyDataset(
        val_df, dataset_root, transform=val_transform, preprocessor=preprocessor
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = EfficientNetB3EDL(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, Config)
    trainer.train()
    
    # Plot results
    trainer.plot_training_history()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, 'final_model.pth'))
    print(f"\nFinal model saved to {Config.CHECKPOINT_DIR}/final_model.pth")


# Call main() directly - this script is meant to be exec'd in notebooks
main()
