"""
Professional Image Preprocessing Pipeline for Digit Recognition
Handles:
- Any image size (1080p, 2K, 4K, etc.)
- Any image format (PNG, JPG, JPEG, WEBP, etc.)
- Canvas drawings
- Photo uploads
- Noise removal
- Digit extraction
- Background removal
"""

import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import Tuple, Optional, Dict

class DigitImageProcessor:
    """
    Process digit images for MNIST-compatible predictions
    Input: Any size image (canvas or photo)
    Output: 28x28 grayscale normalized image
    """
    
    def __init__(self, debug: bool = True):
        """
        Args:
            debug: Save intermediate processing steps to debug_images/
        """
        self.debug = debug
        self.step_counter = 0
    
    def _save_debug_image(self, img: np.ndarray, name: str, normalize: bool = True):
        """Save image for debugging"""
        if not self.debug:
            return
        
        # Convert to 0-255 range if normalized
        if img.dtype == np.float32 or img.dtype == np.float64:
            if normalize and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        path = f"debug_images/{self.step_counter:02d}_{name}.png"
        if len(img.shape) == 2:
            cv2.imwrite(path, img)
        else:
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        self.step_counter += 1
    
    def load_from_base64(self, base64_image: str) -> np.ndarray:
        """Load image from base64 (canvas drawing)"""
        # Remove data URL prefix if present
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB
        if img.mode in ['RGBA', 'LA', 'P']:
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img, dtype=np.uint8)
    
    def load_from_file(self, filepath: str) -> np.ndarray:
        """Load image from file path"""
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"Cannot read image: {filepath}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def load_from_bytes(self, img_bytes: bytes) -> np.ndarray:
        """Load image from raw bytes with EXIF orientation handling"""
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS
        
        img = Image.open(io.BytesIO(img_bytes))
        
        # Handle EXIF orientation for phone images
        try:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "Orientation":
                        # 1 = Normal, 3 = 180°, 6 = 90° CW, 8 = 90° CCW
                        if value == 3:
                            img = img.rotate(180, expand=True)
                        elif value == 6:
                            img = img.rotate(270, expand=True)
                        elif value == 8:
                            img = img.rotate(90, expand=True)
                        break
        except:
            pass  # No EXIF data or error reading it
        
        # Convert mode
        if img.mode not in ['RGB', 'L']:
            img = img.convert('RGB')
        elif img.mode == 'L':
            # If grayscale, convert to RGB for consistency
            img = img.convert('RGB')
        
        result = np.array(img, dtype=np.uint8)
        self._save_debug_image(result, "00a_loaded_with_exif", normalize=False)
        return result
    
    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(img.shape) == 3:
            # RGB to grayscale using standard formula
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        self._save_debug_image(gray, "01_grayscale")
        return gray
    
    def denoise(self, img: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Remove noise from image
        Methods: bilateral, morphological, nlm (non-local means)
        """
        if method == 'bilateral':
            # Bilateral filter: preserves edges while removing noise
            # Phone images often have JPEG artifacts, so use stronger settings
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
            # Apply again for phone images (they need more denoising)
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        elif method == 'morphological':
            # Morphological operations to remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            denoised = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        elif method == 'nlm':
            # Non-local means denoising (slower but better)
            denoised = cv2.fastNlMeansDenoising(img, h=10)
        else:
            denoised = img
        
        self._save_debug_image(denoised, "02_denoised", normalize=False)
        return denoised
    
    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance contrast for phone images with poor lighting
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        self._save_debug_image(enhanced, "02b_enhanced_contrast", normalize=False)
        return enhanced
    
    def binarize(self, img: np.ndarray, threshold: Optional[int] = None) -> np.ndarray:
        """
        Convert to binary (black and white only)
        Args:
            threshold: If None, use Otsu's automatic thresholding
        """
        if threshold is None:
            # Otsu's automatic thresholding
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            
        # CRITICAL: Ensure digit is white and background is black for cv2.findNonZero to work
        # If the mean pixel value is > 127, it means the background is mostly white
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        self._save_debug_image(binary, "03_binary")
        return binary
    
    def find_digit_bbox(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Find bounding box of digit using smart contour detection"""
        # Find all contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, 0, img.shape[1], img.shape[0]
            
        # Filter and score contours
        valid_contours = []
        img_h, img_w = img.shape
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            
            # Filter out tiny noise
            if area < 50 or w < 10 or h < 10:
                continue
                
            # Filter out extreme aspect ratios (like long lines/shadows)
            aspect_ratio = w / float(h)
            if aspect_ratio > 3.0 or aspect_ratio < 0.1:
                continue
                
            # Filter out contours that are just edge shadows
            # (e.g. at the very top/bottom edge and very wide)
            if (y < 10 and w > img_w * 0.5) or (y + h > img_h - 10 and w > img_w * 0.5):
                continue
                
            valid_contours.append(c)
            
        if not valid_contours:
            # Fallback if everything was filtered
            valid_contours = contours
            
        # Find the largest valid contour
        best_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Add dynamic padding based on digit size
        padding = max(10, int(max(w, h) * 0.15))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        return x, y, w, h
    
    def extract_digit(self, img: np.ndarray) -> np.ndarray:
        """Extract digit region and center it"""
        x, y, w, h = self.find_digit_bbox(img)
        digit_img = img[y:y+h, x:x+w].copy()
        
        self._save_debug_image(digit_img, "04_extracted")
        return digit_img
    
    def resize_and_pad(self, img: np.ndarray, size: int = 28) -> np.ndarray:
        """
        Resize image to 28x28 while maintaining aspect ratio
        Pad with background color to fill
        """
        h, w = img.shape[:2]
        
        # Calculate scaling to fit in 28x28 (with padding)
        max_dim = max(h, w)
        scale = (size - 2) / max_dim  # Leave 1 pixel padding on all sides
        
        # Resize
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create padded image with black background (0)
        # Since we inverted earlier if needed, background is guaranteed to be black
        padded = np.full((size, size), 0, dtype=np.uint8)
        
        # Center the resized image
        y_offset = (size - new_h) // 2
        x_offset = (size - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        self._save_debug_image(padded, "05_resized_padded")
        return padded
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize to 0-1 range"""
        normalized = img.astype(np.float32) / 255.0
        self._save_debug_image(normalized, "06_normalized")
        return normalized
    

    
    def process(self, 
                input_data,
                input_type: str = 'base64',
                denoise_method: str = 'bilateral',
                debug: bool = False,
                is_phone_image: bool = False) -> np.ndarray:
        """
        Complete processing pipeline
        
        Args:
            input_data: Base64 string, file path, or bytes
            input_type: 'base64', 'filepath', or 'bytes'
            denoise_method: 'bilateral', 'morphological', 'nlm', or None
            debug: Enable debug image saving
            is_phone_image: Apply extra enhancement for phone photos
        
        Returns:
            Processed image (1D array if output_flat=True)
        """
        self.debug = debug
        self.step_counter = 0
        
        print(f"\n{'='*70}")
        print("[PROCESSOR] Starting image processing pipeline")
        print(f"[INFO] Phone image: {is_phone_image}")
        print(f"{'='*70}")
        
        # Load image
        print("[STEP 1/9] Loading image...")
        if input_type == 'base64':
            img = self.load_from_base64(input_data)
        elif input_type == 'filepath':
            img = self.load_from_file(input_data)
        elif input_type == 'bytes':
            img = self.load_from_bytes(input_data)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        print(f"  Shape: {img.shape}, dtype: {img.dtype}")
        self._save_debug_image(img, "00_original", normalize=False)
        
        # Convert to grayscale
        print("[STEP 2/9] Converting to grayscale...")
        img = self.to_grayscale(img)
        
        # Enhance contrast if phone image
        if is_phone_image:
            print("[STEP 3/9] Enhancing contrast (phone image)...")
            img = self.enhance_contrast(img)
        
        # Denoise
        step_num = 4 if is_phone_image else 3
        print(f"[STEP {step_num}/9] Denoising ({denoise_method})...")
        if denoise_method:
            img = self.denoise(img, denoise_method)
        
        # Binarize
        step_num = 5 if is_phone_image else 4
        print(f"[STEP {step_num}/9] Binarizing image...")
        img = self.binarize(img)
        
        # Extract digit
        step_num = 6 if is_phone_image else 5
        print(f"[STEP {step_num}/9] Extracting digit region...")
        img = self.extract_digit(img)
        
        # Resize and pad
        step_num = 7 if is_phone_image else 6
        print(f"[STEP {step_num}/9] Resizing to 28x28...")
        img = self.resize_and_pad(img, size=28)
        
        # Normalize
        step_num = 8 if is_phone_image else 7
        print(f"[STEP {step_num}/8] Normalizing pixels...")
        img = self.normalize(img)
        
        # Save final image for debugging
        final_viz = (img * 255).astype(np.uint8)
        final_path = 'debug_images/99_final_to_model.png'
        Image.fromarray(final_viz).save(final_path)
        print(f"[DEBUG] Final image saved: {final_path}")
        
        print(f"\n[FINAL] Output shape: {img.shape}")
        print(f"[FINAL] Min: {np.min(img):.4f}, Max: {np.max(img):.4f}, Mean: {np.mean(img):.4f}")
        print(f"{'='*70}\n")
        
        return img


# ============================================
# QUICK TEST
# ============================================
if __name__ == "__main__":
    print("✓ Image processor module loaded successfully")
    print("  Use: processor = DigitImageProcessor()")
    print("  Then: processed_image = processor.process(input_data, 'base64'/'filepath'/'bytes')")
