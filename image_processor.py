"""
EyeShield: Image Preprocessing and Caching Module
Foundational image processing utilities for Diabetic Retinopathy classification

This module provides:
- ImagePreprocessor: High-quality fundus image preprocessing (DICOM + standard formats)
- ImageCacheManager: Efficient disk caching for preprocessing speed improvement
"""

import os
import numpy as np
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import pydicom


# ==================== IMAGE PREPROCESSING ====================
class ImagePreprocessor:
    """Image preprocessing for fundus images - High-quality, format-agnostic"""
    
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
        
        Args:
            image_path: Path to the image file
            
        Returns:
            img_normalized: Numpy array (H, W, 3) normalized to [0, 1]
            
        Raises:
            ValueError: If file cannot be read or processed
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
        
        # Resize with high-quality LANCZOS4 interpolation
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0,1] for model input
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    
    def assess_image_quality(self, preprocessed_img, blur_threshold=70, 
                            brightness_low=30, brightness_high=220, entropy_high=7.5):
        """
        Assesses the quality of a preprocessed fundus image using heuristic criteria.
        
        Args:
            preprocessed_img: Preprocessed image normalized to [0, 1]
            blur_threshold: Laplacian variance threshold for blur detection
            brightness_low: Minimum acceptable mean brightness
            brightness_high: Maximum acceptable mean brightness
            entropy_high: Maximum acceptable entropy
            
        Returns:
            quality_score: Image quality metric (0-1)
            quality_result: Textual quality assessment
            quality_info: Detailed quality metrics dictionary
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
            assess_quality: Whether to assess quality (default: True)
            
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
        Preprocess multiple images efficiently
        
        Args:
            image_paths: List of image paths
            verbose: Print progress with tqdm (default: True)
            
        Returns:
            preprocessed_images: List of preprocessed images
            quality_scores: List of quality scores
            quality_info_list: List of quality info dicts
        """
        preprocessed = []
        quality_scores = []
        quality_info_list = []
        
        iterator = tqdm(image_paths, desc='Preprocessing images') if verbose else image_paths
        
        for img_path in iterator:
            img, quality, info = self.preprocess(img_path, assess_quality=True)
            
            if img is not None:
                preprocessed.append(img)
                quality_scores.append(quality)
                quality_info_list.append(info)
        
        return preprocessed, quality_scores, quality_info_list


# ==================== IMAGE CACHE MANAGER ====================
class ImageCacheManager:
    """
    Manage image preprocessing and caching for fast training.
    Provides 10x speedup by caching preprocessed images to disk.
    """
    
    def __init__(self, cache_dir='./image_cache', preprocessor=None):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cached images
            preprocessor: ImagePreprocessor instance for preprocessing
        """
        self.cache_dir = cache_dir
        self.preprocessor = preprocessor
        os.makedirs(cache_dir, exist_ok=True)
        self.metadata_file = os.path.join(cache_dir, 'metadata.pkl')
    
    def get_cache_path(self, image_filename):
        """
        Get the cache file path for an image
        
        Args:
            image_filename: Original image filename
            
        Returns:
            str: Path to cached .npy file
        """
        # Use a flattened filename so cache files always live directly under cache_dir
        # and work consistently across OS path separator conventions.
        safe_filename = str(image_filename).replace('/', '__').replace('\\', '__')
        return os.path.join(self.cache_dir, f"{safe_filename}.npy")

    def _get_legacy_cache_path(self, image_filename):
        """Legacy cache path used by earlier versions (kept for backward compatibility)."""
        return os.path.join(self.cache_dir, f"{image_filename}.npy")

    def cache_exists(self, image_filename):
        """Check whether cache exists in either new or legacy naming format."""
        return os.path.exists(self.get_cache_path(image_filename)) or os.path.exists(self._get_legacy_cache_path(image_filename))
    
    def preprocess_and_cache(self, df, dataset_root, force_reprocess=False):
        """
        Preprocess all images and cache them to disk.
        
        One-time operation; skips already-cached images on re-runs unless force_reprocess=True.
        Cached images are stored as .npy binary files for fast I/O.
        
        Args:
            df: DataFrame with 'image_path' column (relative paths from dataset_root)
            dataset_root: Root directory containing images
            force_reprocess: If True, reprocess even if cached (default: False)
            
        Returns:
            bool: True if all images cached successfully, False if any failed
        """
        print(f"Preprocessing {len(df)} images to cache...")
        print(f"Cache location: {self.cache_dir}\n")
        
        cached_count = 0
        new_count = 0
        failed_images = []
        cache_metadata = {}
        
        pbar = tqdm(total=len(df), desc='Caching images', unit='img')
        
        for idx, row in df.iterrows():
            img_path = os.path.join(dataset_root, row['image_path'])
            cache_path = self.get_cache_path(row['image_path'])
            legacy_cache_path = self._get_legacy_cache_path(row['image_path'])
            
            # If already cached and not forcing reprocessing, skip
            if (os.path.exists(cache_path) or os.path.exists(legacy_cache_path)) and not force_reprocess:
                cached_count += 1
                cache_metadata[row['image_path']] = 'cached'
                pbar.update(1)
                continue
            
            try:
                # Preprocess image (skip quality check for speed)
                preprocessed_img, quality_score, quality_info = self.preprocessor.preprocess(
                    img_path, assess_quality=False
                )
                
                if preprocessed_img is None:
                    failed_images.append((row['image_path'], 'Preprocessing returned None'))
                    cache_metadata[row['image_path']] = 'failed'
                    pbar.update(1)
                    continue
                
                # Save as NumPy binary (.npy) for fast I/O
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, preprocessed_img.astype(np.float32))
                new_count += 1
                cache_metadata[row['image_path']] = 'new'
                
            except Exception as e:
                failed_images.append((row['image_path'], str(e)))
                cache_metadata[row['image_path']] = 'failed'
            
            pbar.update(1)
        
        pbar.close()
        
        # Save metadata for tracking
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(cache_metadata, f)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"CACHING COMPLETE")
        print(f"{'='*80}")
        print(f"Total images:      {len(df)}")
        print(f"  ✓ Already cached: {cached_count}")
        print(f"  ✓ Newly cached:   {new_count}")
        print(f"  ✗ Failed:         {len(failed_images)}")
        print(f"\nCache size:        {self._get_cache_size_gb():.2f} GB")
        print(f"{'='*80}\n")
        
        if failed_images:
            print("Failed images (showing first 10):")
            for img_path, error in failed_images[:10]:
                print(f"  - {img_path}: {error}")
            if len(failed_images) > 10:
                print(f"  ... and {len(failed_images) - 10} more")
            print()
        
        return len(failed_images) == 0
    
    def load_cached_image(self, image_filename):
        """
        Load preprocessed image from cache
        
        Args:
            image_filename: Original image filename
            
        Returns:
            numpy.ndarray: Preprocessed image (H, W, 3) normalized to [0, 1]
            
        Raises:
            FileNotFoundError: If cached image doesn't exist
        """
        cache_path = self.get_cache_path(image_filename)
        legacy_cache_path = self._get_legacy_cache_path(image_filename)

        if os.path.exists(cache_path):
            return np.load(cache_path)

        if os.path.exists(legacy_cache_path):
            return np.load(legacy_cache_path)

        raise FileNotFoundError(
            f"Cached image not found in either format:\n"
            f"  - New: {cache_path}\n"
            f"  - Legacy: {legacy_cache_path}"
        )
    
    def clear_cache(self):
        """Clear all cached images from disk"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            print(f"✓ Cache cleared: {self.cache_dir}")
    
    def get_cache_stats(self):
        """
        Get statistics about cached images
        
        Returns:
            dict: Dictionary with cache statistics
        """
        stats = {
            'total_images': 0,
            'cached_images': 0,
            'failed_images': 0,
            'total_size_gb': self._get_cache_size_gb()
        }
        
        # Load metadata if available
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            for status in metadata.values():
                if status == 'cached' or status == 'new':
                    stats['cached_images'] += 1
                elif status == 'failed':
                    stats['failed_images'] += 1
                stats['total_images'] += 1
        
        return stats
    
    def _get_cache_size_gb(self):
        """
        Get total cache size in GB
        
        Returns:
            float: Cache size in gigabytes
        """
        total_size = 0
        if os.path.exists(self.cache_dir):
            for root, _, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith('.npy'):  # Only count .npy cache files
                        file_path = os.path.join(root, file)
                        if os.path.isfile(file_path):
                            total_size += os.path.getsize(file_path)
        return total_size / (1024**3)


if __name__ == "__main__":
    """
    Example usage of ImagePreprocessor and ImageCacheManager
    """
    print("ImagePreprocessor and ImageCacheManager Module")
    print("Import these classes into your training scripts\n")
    print("Example:")
    print("  from image_processor import ImagePreprocessor, ImageCacheManager")
    print("  preprocessor = ImagePreprocessor(target_size=(512, 512))")
    print("  cache_manager = ImageCacheManager(preprocessor=preprocessor)")
