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
import pickle
import shutil
from pathlib import Path
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import pydicom
from torch.utils.data import WeightedRandomSampler

# Image preprocessing and caching (separated module for efficiency)
from image_processor import ImagePreprocessor, ImageCacheManager


def apply_image_cache_manager_compatibility_patch():
    """
    Patch ImageCacheManager at runtime for backward compatibility.

    This handles older image_processor.py versions that:
    - Build nested cache paths like /content/image_cache/train/0/x.jpg.npy
      (which can fail when parent dirs don't exist)
    - Do not provide cache_exists()
    - Cannot load legacy/new cache naming formats interchangeably
    """

    def _safe_get_cache_path(self, image_filename):
        safe_filename = str(image_filename).replace('/', '__').replace('\\', '__')
        return os.path.join(self.cache_dir, f"{safe_filename}.npy")

    def _legacy_get_cache_path(self, image_filename):
        return os.path.join(self.cache_dir, f"{image_filename}.npy")

    def _cache_exists(self, image_filename):
        return os.path.exists(_safe_get_cache_path(self, image_filename)) or os.path.exists(
            _legacy_get_cache_path(self, image_filename)
        )

    def _load_cached_image(self, image_filename):
        new_path = _safe_get_cache_path(self, image_filename)
        legacy_path = _legacy_get_cache_path(self, image_filename)

        if os.path.exists(new_path):
            img = np.load(new_path)
        elif os.path.exists(legacy_path):
            img = np.load(legacy_path)
        else:
            raise FileNotFoundError(
                f"Cached image not found in either format:\n"
                f"  - New: {new_path}\n"
                f"  - Legacy: {legacy_path}"
            )

        # Support both float32 cache and uint8 cache formats.
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)

    def _get_cache_size_gb(self):
        total_size = 0
        if os.path.exists(self.cache_dir):
            for root, _, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        if os.path.isfile(file_path):
                            total_size += os.path.getsize(file_path)
        return total_size / (1024**3)

    # Patch unconditionally so behavior is consistent across versions.
    ImageCacheManager.get_cache_path = _safe_get_cache_path
    ImageCacheManager._get_legacy_cache_path = _legacy_get_cache_path
    ImageCacheManager.cache_exists = _cache_exists
    ImageCacheManager.load_cached_image = _load_cached_image
    ImageCacheManager._get_cache_size_gb = _get_cache_size_gb


apply_image_cache_manager_compatibility_patch()

# GPU/Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==================== CONFIGURATION ====================
class Config:
    """Configuration for training"""
    
    # Image Preprocessing
    TARGET_IMAGE_SIZE = (512, 512)  # Your preferred size for training
    QUALITY_CHECK = False            # Assess image quality (disabled for speed/memory)
    
    # Dataset paths
    KAGGLE_DATASET_PATH = '/kaggle/input'
    COLAB_DRIVE_PATH = '/content/drive/MyDrive'
    MAX_DATASET_SIZE = 60000  # Limit dataset to N images for faster training (set to None for all)
    
    # Model parameters
    NUM_CLASSES = 5  # Grade 0-4: No DR, Mild, Moderate, Severe, Proliferative
    INPUT_SIZE = (512, 512)
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # EDL parameters
    EDL_UNCERTAINTY_THRESHOLD = 0.3
    KL_WEIGHT = 0.1
    ANNEALING_START = 10
    ANNEALING_STEP = 1.0
    
    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Early stopping (patience for validation loss non-improvement)
    EARLY_STOPPING_PATIENCE = 15
    
    # Checkpoint and logging
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    SAVE_INTERVAL = 5
    MAX_CHECKPOINTS = 3  # Keep only top 3 checkpoints
    
    # Augmentation
    AUGMENT = True
    RANDOM_SEED = 42
    
    # Data loading (auto-detected based on device)
    # NUM_WORKERS will be set based on platform (0 for Colab, 4 for local)
    NUM_WORKERS = 0 if torch.cuda.is_available() and 'COLAB_RELEASE_TAG' in os.environ else 4


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
            
            # Check if file exists - fail explicitly instead of silently using blank tensor
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found at: {img_path}\n"
                                       f"  Expected path: {img_path}\n"
                                       f"  Please verify dataset paths in CSV are correct relative to dataset root.")
            
            img, quality_score, quality_info = self.preprocessor.preprocess(
                img_path, assess_quality=Config.QUALITY_CHECK
            )
            
            if img is None:
                raise ValueError(f"Failed to preprocess image: {img_path}")
            
            # Convert to PIL for transforms
            pil = Image.fromarray((img * 255).astype(np.uint8))
            
            if self.transform is not None:
                pil = self.transform(pil)
            
            label = int(row['diagnosis'])
            return pil, label
        
        except Exception as e:
            print(f"❌ Error loading batch item {idx}: {e}")
            raise  # Re-raise to fail training explicitly


# ==================== CACHED DATASET (FAST - LOADS FROM CACHE) ====================
class CachedDiabeticRetinopathyDataset(Dataset):
    """
    Dataset that loads preprocessed images from cache.
    This is 10x faster than preprocessing on-the-fly.
    """
    
    def __init__(self, df, cache_manager, transform=None):
        """
        Args:
            df: DataFrame with columns ['image_path', 'diagnosis']
            cache_manager: ImageCacheManager instance with cached images
            transform: Optional transforms to apply (augmentation)
        """
        self.df = df
        self.cache_manager = cache_manager
        self.transform = transform
        self.classes = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # Load from cache (instant - no preprocessing needed)
            img = self.cache_manager.load_cached_image(row['image_path'])
            
            # img is already normalized [0, 1] from preprocessing
            # Convert to PIL for transforms
            pil = Image.fromarray((img * 255).astype(np.uint8))
            
            if self.transform is not None:
                pil = self.transform(pil)
            
            label = int(row['diagnosis'])
            return pil, label
        
        except Exception as e:
            print(f"❌ Error loading cached image {idx}: {e}")
            raise


def get_data_transforms(augment=True):
    """
    Data augmentation and normalization for fundus images.
    
    IMPORTANT: Fundus images have specific anatomical orientation.
    Do NOT use RandomHorizontalFlip or RandomVerticalFlip as these
    change the eye structure and lead to poor generalization.
    """
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        # ✓ CORRECT augmentation for fundus images
        train_transform = transforms.Compose([
            # ✓ Small rotations (±10°) - fundus images can be rotated slightly
            transforms.RandomRotation(10, fill=0),
            
            # ✓ Small translations - optical disk might be slightly off-center
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                fill=0
            ),
            
            # ✓ Color jittering - accounts for different imaging equipment
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            
            # ✓ Subtle blur - accounts for slight focus variations
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0)
            ),
            
            # ❌ DO NOT USE - corrupts fundus anatomy:
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    val_transform = transforms.Compose([
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
        
        # Evidence layer with ReLU activation and batch normalization
        self.evidence_layer = nn.Sequential(
            nn.Linear(input_features, 512),      # Expanded
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),                 # New intermediate
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)          # No BatchNorm here!
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
def calculate_class_weights(labels, num_classes):
    """
    Calculate class weights to handle imbalanced data.
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
    
    Returns:
        torch.Tensor: Class weights tensor
    
    Raises:
        ValueError: If any class has 0 samples
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Check for missing classes
    empty_classes = np.where(class_counts == 0)[0]
    if len(empty_classes) > 0:
        raise ValueError(f"Classes {list(empty_classes)} have 0 samples in training set. "
                        f"This will cause training failures. Use stratified splitting to ensure all classes are present.")
    
    # Weight = total_samples / (num_classes * class_count)
    # This is the effective number of samples method
    class_weights = total_samples / (num_classes * (class_counts + 1e-6))
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
    
    return torch.tensor(class_weights, dtype=torch.float32)


def get_weighted_sampler(df, num_classes):
    """
    Create a WeightedRandomSampler for imbalanced classification.
    
    Args:
        df: DataFrame with 'diagnosis' column
        num_classes: Number of classes
    
    Returns:
        WeightedRandomSampler: Sampler that balances classes
    """
    # Get class weights
    class_weights = calculate_class_weights(df['diagnosis'].values, num_classes)
    
    # Assign weight to each sample based on its class
    sample_weights = class_weights[df['diagnosis'].values].numpy()
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(df),
        replacement=True
    )
    
    return sampler, class_weights


class EvidentialLoss(nn.Module):
    """Type-II Maximum Likelihood EDL Loss with Class Weights"""
    
    def __init__(self, num_classes, kl_weight=0.1, class_weights=None):
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        
        # Register class weights as buffer (will be moved to device with model)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', torch.ones(num_classes))
    
    def forward(self, evidence, targets, epoch, annealing_step):
        """Compute EDL loss with annealing KL regularization and class weights"""
        
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
        
        # Apply class weights
        class_weight_targets = self.class_weights[targets].unsqueeze(1)
        nll_loss = nll_loss * class_weight_targets
        
        # KL regularization with annealing
        lam = min(1.0, epoch / annealing_step) if epoch < annealing_step else 1.0
        
        kl_alpha = (alpha - 1) * (1 - targets_one_hot)
        kl_loss = lam * torch.sum(
            torch.lgamma(torch.sum(alpha, dim=1, keepdim=True)) - 
            torch.sum(torch.lgamma(alpha), dim=1, keepdim=True) +
            torch.sum(kl_alpha * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=True),
            dim=1, keepdim=True
        )
        
        # Apply class weights to KL loss as well
        kl_loss = kl_loss * class_weight_targets
        
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
        # Use view(-1) instead of squeeze to avoid shape issues with batch_size=1
        self.uncertainties.extend(output['uncertainty'].view(-1).cpu().detach().numpy())
        self.confidences.extend(output['confidence'].view(-1).cpu().detach().numpy())
    
    def compute(self):
        """Compute all metrics"""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        uncertainties = np.array(self.uncertainties)
        confidences = np.array(self.confidences)
        
        accuracy = accuracy_score(targets, preds)
        
        # Macro F1 Score (recommended for imbalanced classification)
        macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
        
        # Weighted F1 Score
        weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
        
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
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'ece': ece,
            'mean_uncertainty': np.mean(uncertainties),
            'mean_confidence': np.mean(confidences),
            'confusion_matrix': confusion_matrix(targets, preds, labels=range(self.num_classes))
        }


# ==================== TRAINING ====================
class Trainer:
    """Training loop for EDL model"""
    
    def __init__(self, model, train_loader, val_loader, config, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss and optimizer
        self.criterion = EvidentialLoss(
            config.NUM_CLASSES, 
            kl_weight=config.KL_WEIGHT,
            class_weights=class_weights
        )
        self.criterion = self.criterion.to(self.device)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics
        self.metrics = EDLMetrics(config.NUM_CLASSES)
        
        # History
        self.history = {
            'train_loss': [], 'train_nll': [], 'train_kl': [], 'train_acc': [], 'train_macro_f1': [],
            'val_loss': [], 'val_nll': [], 'val_kl': [], 'val_acc': [], 'val_macro_f1': [], 'val_ece': []
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
            'accuracy': train_metrics['accuracy'],
            'macro_f1': train_metrics['macro_f1']
        }
    
    def validate(self, epoch):
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
                    evidence, targets, epoch, 
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
            'macro_f1': val_metrics['macro_f1'],
            'ece': val_metrics['ece'],
            'confusion_matrix': val_metrics['confusion_matrix']
        }
    
    def train(self):
        """Full training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.EARLY_STOPPING_PATIENCE  # Now configurable from Config
        
        print("\n" + "="*80)
        print("Starting Training: EfficientNet-B3 + EDL for DR Classification")
        print("Using Your Image Preprocessor")
        print("="*80 + "\n")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_nll'].append(train_metrics['nll'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_macro_f1'].append(train_metrics['macro_f1'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_nll'].append(val_metrics['nll'])
            self.history['val_kl'].append(val_metrics['kl'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_macro_f1'].append(val_metrics['macro_f1'])
            self.history['val_ece'].append(val_metrics['ece'])
            
            # Update scheduler based on validation loss
            self.scheduler.step(val_metrics['loss'])
            
            # Logging
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train Macro F1: {train_metrics['macro_f1']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f} | ECE: {val_metrics['ece']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch, val_metrics['macro_f1'])
            
            # Early stopping based on validation loss (more stable than accuracy for imbalanced data)
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_best_model(epoch, val_metrics)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (validation loss did not improve)")
                    break
            
            # Clear cache to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
    
    def save_checkpoint(self, epoch, metric_value):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metric': metric_value
        }
        path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'checkpoint_epoch_{epoch+1}_f1_{metric_value:.4f}.pt'
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
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
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
        
        # Macro F1 Score (recommended for imbalanced data)
        axes[1, 1].plot(self.history['train_macro_f1'], label='Train')
        axes[1, 1].plot(self.history['val_macro_f1'], label='Val')
        axes[1, 1].set_title('Macro F1 Score (Primary Metric for Imbalanced Data)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Macro F1')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # KL Loss
        axes[2, 0].plot(self.history['train_kl'], label='Train KL')
        axes[2, 0].plot(self.history['val_kl'], label='Val KL')
        axes[2, 0].set_title('KL Regularization Loss')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('KL Loss')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # ECE (Calibration)
        axes[2, 1].plot(self.history['val_ece'], label='Calibration Error')
        axes[2, 1].set_title('Expected Calibration Error')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('ECE')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.LOG_DIR, 'training_history.png'), dpi=300)
        print(f"Training history plot saved to {self.config.LOG_DIR}")
        plt.show()


# ==================== MAIN ====================
def visualize_class_distribution(train_df, val_df, test_df, class_weights, log_dir):
    """
    Visualize class distribution before and after balancing.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        class_weights: Calculated class weights
        log_dir: Directory to save visualizations
    """
    os.makedirs(log_dir, exist_ok=True)
    
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Training set class distribution
    train_counts = train_df['diagnosis'].value_counts().sort_index()
    axes[0, 0].bar(range(len(train_counts)), train_counts.values, color='steelblue')
    axes[0, 0].set_xticks(range(len(train_counts)))
    axes[0, 0].set_xticklabels([class_names[i] for i in train_counts.index])
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Original Training Set Distribution')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(train_counts.values):
        axes[0, 0].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Validation set class distribution
    val_counts = val_df['diagnosis'].value_counts().sort_index()
    axes[0, 1].bar(range(len(val_counts)), val_counts.values, color='darkorange')
    axes[0, 1].set_xticks(range(len(val_counts)))
    axes[0, 1].set_xticklabels([class_names[i] for i in val_counts.index])
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Validation Set Distribution')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(val_counts.values):
        axes[0, 1].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 3. Class weights used for balancing
    axes[1, 0].bar(range(len(class_weights)), class_weights.numpy(), color='darkgreen')
    axes[1, 0].set_xticks(range(len(class_weights)))
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Class Weights for Balancing')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(class_weights.numpy()):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Imbalance ratio (percentage)
    total_train = len(train_df)
    percentages = [(train_counts.get(i, 0) / total_train * 100) for i in range(len(class_names))]
    axes[1, 1].bar(range(len(percentages)), percentages, color='crimson')
    axes[1, 1].set_xticks(range(len(percentages)))
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_title('Class Distribution (% of Training Set)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(percentages):
        axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(log_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Class distribution visualization saved to {save_path}")
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION STATISTICS")
    print("="*80)
    print(f"\nTraining Set ({len(train_df)} total):")
    for i, count in enumerate(train_counts):
        pct = (count / len(train_df) * 100)
        weight = class_weights[i].item()
        print(f"  {class_names[i]:15s}: {count:5d} samples ({pct:5.1f}%) | Weight: {weight:.4f}")
    
    print(f"\nValidation Set ({len(val_df)} total):")
    for i in range(len(class_names)):
        count = val_counts.get(i, 0)
        pct = (count / len(val_df) * 100)
        print(f"  {class_names[i]:15s}: {count:5d} samples ({pct:5.1f}%)")
    
    print(f"\nTest Set ({len(test_df)} total):")
    test_counts = test_df['diagnosis'].value_counts().sort_index()
    for i in range(len(class_names)):
        count = test_counts.get(i, 0)
        pct = (count / len(test_df) * 100) if len(test_df) > 0 else 0
        print(f"  {class_names[i]:15s}: {count:5d} samples ({pct:5.1f}%)")
    
    # Calculate imbalance ratio
    max_count = train_counts.max()
    min_count = train_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}")
    print("="*80 + "\n")


def resolve_dataset_root(df):
    """
    Resolve a dataset root that matches relative paths stored in labels.csv.

    Priority:
      1) /content/dataset/data_root.txt (saved by the notebook CSV step)
      2) kagglehub download root (+ common nested subfolders)
      3) common Kaggle input paths
      4) bounded recursive search fallback
    """
    if len(df) == 0:
        raise ValueError("Cannot resolve dataset root: dataframe is empty.")

    sample_rel_path = str(df.iloc[0]['image_path']).lstrip('/\\')
    candidate_roots = []

    # 1) Prefer exact root saved during CSV creation
    data_root_file = '/content/dataset/data_root.txt'
    if os.path.exists(data_root_file):
        try:
            with open(data_root_file, 'r') as f:
                saved_root = f.read().strip()
            if saved_root:
                candidate_roots.append(saved_root)
        except Exception as e:
            print(f"⚠ Could not read {data_root_file}: {e}")

    # 2) kagglehub + common nested layouts
    try:
        import kagglehub
        kagglehub_root = kagglehub.dataset_download("ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy")
        candidate_roots.extend([
            kagglehub_root,
            os.path.join(kagglehub_root, 'dr_unified_v2'),
            os.path.join(kagglehub_root, 'dr_unified_v2', 'dr_unified_v2'),
        ])
    except Exception as e:
        print(f"⚠ Could not resolve kagglehub root automatically: {e}")

    # 3) Common Kaggle mounted paths
    candidate_roots.extend([
        '/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy',
        '/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/dr_unified_v2',
        '/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/dr_unified_v2/dr_unified_v2',
    ])

    # De-duplicate while preserving order
    unique_roots = []
    seen = set()
    for root in candidate_roots:
        if not root:
            continue
        norm_root = os.path.normpath(root)
        if norm_root not in seen:
            seen.add(norm_root)
            unique_roots.append(norm_root)

    print("Resolving dataset root...")
    for root in unique_roots:
        if os.path.exists(os.path.join(root, sample_rel_path)):
            print(f"✓ Dataset root resolved: {root}")
            return root

    # 4) Fallback search (bounded depth)
    search_bases = ['/kaggle/input', '/root/.cache/kagglehub']
    for base in search_bases:
        if not os.path.isdir(base):
            continue
        print(f"  Searching recursively under: {base}")
        for current_root, dirs, _ in os.walk(base):
            depth = os.path.relpath(current_root, base).count(os.sep)
            if depth > 5:
                dirs[:] = []
                continue
            if os.path.exists(os.path.join(current_root, sample_rel_path)):
                print(f"✓ Dataset root discovered via search: {current_root}")
                return current_root

    raise FileNotFoundError(
        "Could not resolve dataset root for CSV paths.\n"
        f"Sample relative path: {sample_rel_path}\n"
        "Expected this file under one of the dataset roots, but it was not found.\n"
        "Please re-run 'Prepare Dataset CSV' so data_root.txt is regenerated correctly."
    )


def filter_to_cached_images(df, cache_manager, split_name):
    """Keep only rows with existing cached .npy files (new or legacy cache format)."""
    keep_mask = df['image_path'].apply(cache_manager.cache_exists)
    filtered_df = df[keep_mask].reset_index(drop=True)
    dropped = len(df) - len(filtered_df)

    if dropped > 0:
        print(f"⚠ {split_name}: Dropped {dropped} samples missing cached files")
    else:
        print(f"✓ {split_name}: All samples have cached files")

    return filtered_df


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
    csv_path = '/content/dataset/labels.csv'
    
    # Validate CSV exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}\n"
                               f"Please ensure the 'Prepare Dataset CSV' cell has been executed first.")
    
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        raise ValueError(f"Dataset CSV is empty at: {csv_path}")
    
    # Limit dataset size if MAX_DATASET_SIZE is set
    if Config.MAX_DATASET_SIZE is not None and len(df) > Config.MAX_DATASET_SIZE:
        print(f"⚠️  Limiting dataset from {len(df)} → {Config.MAX_DATASET_SIZE} images")
        # Sample uniformly from each class to maintain stratification
        df = df.groupby('diagnosis', group_keys=False).apply(
            lambda x: x.sample(n=max(1, int(Config.MAX_DATASET_SIZE * len(x) / len(df))), random_state=Config.RANDOM_SEED)
        ).reset_index(drop=True)
        
        # If still over limit, randomly sample
        if len(df) > Config.MAX_DATASET_SIZE:
            df = df.sample(n=Config.MAX_DATASET_SIZE, random_state=Config.RANDOM_SEED).reset_index(drop=True)
    
    print(f"✓ Using {len(df)} images for training")
    
    # Resolve dataset root to match relative paths stored in labels.csv
    dataset_root = resolve_dataset_root(df)
    print(f"✓ Loaded {len(df)} images from dataset")
    print(f"  - Dataset root: {dataset_root}")
    print(f"  - Class distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")
    
    # Stratified split data to maintain class distribution
    print("Performing stratified data split...")
    
    # First split: separate test set (stratified)
    train_val_df, test_df = train_test_split(
        df,
        test_size=Config.TEST_RATIO,
        stratify=df['diagnosis'],
        random_state=Config.RANDOM_SEED
    )
    
    # Second split: separate train and val sets (stratified)
    # Adjust val_ratio relative to remaining data
    val_ratio_adjusted = Config.VAL_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df['diagnosis'],
        random_state=Config.RANDOM_SEED
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"✓ Stratified data split complete:")
    print(f"  - Train: {len(train_df)} images")
    print(f"  - Val: {len(val_df)} images")
    print(f"  - Test: {len(test_df)} images\n")
    
    # VALIDATION: Check that all classes are present in training set
    train_classes = set(train_df['diagnosis'].unique())
    expected_classes = set(range(Config.NUM_CLASSES))
    missing_classes = expected_classes - train_classes
    
    if missing_classes:
        print(f"⚠️  WARNING: Missing classes in training set: {sorted(list(missing_classes))}")
        print(f"   This can cause training errors. Consider adjusting data split ratios.")
    else:
        print(f"✓ All {Config.NUM_CLASSES} classes present in training set\n")
    
    # Data transforms
    train_transform, val_transform = get_data_transforms(augment=Config.AUGMENT)
    
    # Calculate class weights for imbalanced data
    print("Calculating class weights for imbalanced data...")
    class_sampler, class_weights = get_weighted_sampler(train_df, Config.NUM_CLASSES)
    print("✓ Class weights calculated:")
    for i, weight in enumerate(class_weights):
        class_name = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'][i]
        print(f"  - Class {i} ({class_name}): {weight:.4f}")
    print()
    
    # Visualize class distribution
    visualize_class_distribution(train_df, val_df, test_df, class_weights, Config.LOG_DIR)
    
    # ==================== CACHE IMAGES (ONE-TIME SETUP) ====================
    print("\n" + "="*80)
    print("PREPROCESSING AND CACHING IMAGES")
    print("This is a one-time operation. Subsequent runs will skip already-cached images.")
    print("="*80 + "\n")
    
    # Use Colab ephemeral storage (doesn't use Google Drive quota)
    cache_dir = '/content/image_cache'
    
    # Initialize cache manager
    cache_manager = ImageCacheManager(cache_dir=cache_dir, preprocessor=preprocessor)
    
    # Cache all images (train + val + test combined)
    all_images_df = pd.concat([train_df, val_df, test_df], ignore_index=True).drop_duplicates(subset=['image_path'])
    cache_ok = cache_manager.preprocess_and_cache(all_images_df, dataset_root, force_reprocess=False)
    if not cache_ok:
        print("⚠ Some images failed during caching; training will continue with available cached images.")

    # Keep only samples that have cache files so DataLoader never hits missing-cache errors
    train_df = filter_to_cached_images(train_df, cache_manager, split_name='Train')
    val_df = filter_to_cached_images(val_df, cache_manager, split_name='Val')

    if len(train_df) == 0:
        raise RuntimeError(
            "No training samples have cached images. "
            "This usually means dataset_root does not match CSV relative paths."
        )
    if len(val_df) == 0:
        raise RuntimeError("No validation samples have cached images. Cannot continue training.")

    # Recalculate class sampler/weights after cache filtering to keep them consistent
    print("Recalculating class weights after cache validation...")
    class_sampler, class_weights = get_weighted_sampler(train_df, Config.NUM_CLASSES)
    
    print("✓ Images cached! Training will now load from cache (10x faster).\n")
    
    # ==================== LOAD CACHED DATASETS ====================
    print("Loading datasets from cache...")
    train_dataset = CachedDiabeticRetinopathyDataset(
        train_df, cache_manager=cache_manager, transform=train_transform
    )
    val_dataset = CachedDiabeticRetinopathyDataset(
        val_df, cache_manager=cache_manager, transform=val_transform
    )
    print(f"✓ Datasets loaded:")
    print(f"  - Train: {len(train_dataset)} images (cached)")
    print(f"  - Val: {len(val_dataset)} images (cached)")
    print(f"  - Expected per-epoch time: ~6 minutes (down from 58 minutes)\n")
    
    # Use WeightedRandomSampler for training to handle class imbalance
    pin_memory = torch.cuda.is_available()  # Only pin memory if GPU available
    
    # DataLoaders remain the same - they'll load from cache instead of preprocessing files
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=class_sampler,  # Use weighted sampler for balanced batches
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory
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
    trainer = Trainer(model, train_loader, val_loader, Config, class_weights=class_weights)
    trainer.train()
    
    # Plot results
    trainer.plot_training_history()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, 'final_model.pth'))
    print(f"\nFinal model saved to {Config.CHECKPOINT_DIR}/final_model.pth")


# Call main() directly - this script is meant to be exec'd in notebooks
main()
