"""
Image Quality Metrics (No Text Reference)
==========================================

These metrics assess technical image quality independent of the prompt.

Metrics:
- BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator (via piq)
- NIQE: Natural Image Quality Evaluator (via piq)
- CLIP-IQA: CLIP-based Image Quality Assessment (via pyiqa)
"""

import numpy as np
import torch

# Try importing piq for BRISQUE and NIQE
try:
    import piq
    PIQ_AVAILABLE = True
    # Check if NIQE is available (it's a class, not a function)
    _piq_niqe = None
except ImportError:
    PIQ_AVAILABLE = False
    print("Warning: piq not available. Install with: pip install piq")

# Try importing pyiqa for CLIP-IQA and NIQE
try:
    import pyiqa
    PYIQA_AVAILABLE = True
    # Pre-load models (lazy loading on first use)
    _clipiqa_model = None
    _niqe_model = None
except ImportError:
    PYIQA_AVAILABLE = False
    print("Warning: pyiqa not available. Install with: pip install pyiqa")

# Fallback imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from skimage.util import img_as_float
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    def img_as_float(img):
        """Fallback implementation"""
        return np.array(img, dtype=np.float64) / 255.0


def _pil_to_tensor(image):
    """Convert PIL image to torch tensor for piq/pyiqa (BCHW format, 0-1 range)."""
    img_array = np.array(image)
    # Handle grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    # Convert HWC to CHW and normalize to 0-1
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    # Add batch dimension
    return img_tensor.unsqueeze(0)


def calculate_brisque_score(image):
    """
    Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score.
    
    Uses the piq library for accurate BRISQUE calculation.
    Lower raw BRISQUE scores indicate better quality (typically 0-100).
    We invert and normalize to 0-100 scale where higher is better.
    
    Args:
        image: PIL Image to evaluate
    
    Returns:
        float: Quality score 0-100 (higher is better)
    """
    if PIQ_AVAILABLE:
        try:
            img_tensor = _pil_to_tensor(image)
            # piq.brisque returns lower scores for better quality (typically 0-100)
            raw_score = piq.brisque(img_tensor, data_range=1.0).item()
            # Invert: 100 - raw_score, clamped to 0-100
            # Typical BRISQUE range is 0-100, lower is better
            quality_score = max(0, min(100, 100 - raw_score))
            return quality_score
        except Exception as e:
            print(f"piq BRISQUE error: {e}, falling back to custom implementation")
    
    # Fallback implementation
    try:
        img_array = np.array(image)
        if not CV2_AVAILABLE:
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            gy, gx = np.gradient(gray)
            sharpness = np.sqrt(gx**2 + gy**2).var()
            contrast = gray.std()
            quality_score = min(100, (sharpness / 50 * 50) + (contrast / 50 * 50))
            return quality_score
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        contrast = gray.std()
        quality_score = min(100, (sharpness / 500 * 50) + (contrast / 50 * 50))
        return quality_score
        
    except Exception as e:
        print(f"BRISQUE calculation error: {e}")
        return 0.0


def calculate_niqe_score(image):
    """
    Calculate NIQE (Natural Image Quality Evaluator) score.
    
    Uses the pyiqa library for accurate NIQE calculation.
    Lower raw NIQE scores indicate better quality (typically 0-10).
    We invert and normalize to 0-100 scale where higher is better.
    
    Args:
        image: PIL Image to evaluate
    
    Returns:
        float: Quality score 0-100 (higher is better)
    """
    global _niqe_model
    
    # Try pyiqa first (more reliable NIQE implementation)
    if PYIQA_AVAILABLE:
        try:
            if _niqe_model is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _niqe_model = pyiqa.create_metric('niqe', device=device)
            
            img_tensor = _pil_to_tensor(image)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            # pyiqa niqe returns lower scores for better quality (typically 0-10)
            raw_score = _niqe_model(img_tensor).item()
            # Invert and scale: NIQE typically ranges 0-10, lower is better
            quality_score = max(0, min(100, (10 - raw_score) * 10))
            return quality_score
        except Exception as e:
            print(f"pyiqa NIQE error: {e}, falling back to custom implementation")
    
    # Fallback implementation
    try:
        img_array = np.array(image)
        if not CV2_AVAILABLE:
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            gy, gx = np.gradient(gray)
            edge_magnitude = np.sqrt(gx**2 + gy**2)
            edge_density = np.sum(edge_magnitude > 20) / edge_magnitude.size
            naturalness_score = min(100, (entropy / 8 * 50) + (edge_density * 5000))
            return naturalness_score
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        naturalness_score = min(100, (entropy / 8 * 50) + (edge_density * 5000))
        return naturalness_score
        
    except Exception as e:
        print(f"NIQE calculation error: {e}")
        return 0.0


def calculate_clip_iqa_score(image):
    """
    Calculate CLIP-IQA score using pyiqa library.
    
    CLIP-IQA uses CLIP embeddings to assess perceptual image quality.
    Returns a score where higher indicates better quality.
    
    Args:
        image: PIL Image to evaluate
    
    Returns:
        float: Quality score 0-100 (higher is better)
    """
    global _clipiqa_model
    
    if PYIQA_AVAILABLE:
        try:
            # Lazy load the model
            if _clipiqa_model is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _clipiqa_model = pyiqa.create_metric('clipiqa', device=device)
            
            img_tensor = _pil_to_tensor(image)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            # pyiqa clipiqa returns scores typically in 0-1 range
            raw_score = _clipiqa_model(img_tensor).item()
            # Scale to 0-100
            quality_score = raw_score * 100
            return min(100, max(0, quality_score))
            
        except Exception as e:
            print(f"pyiqa CLIP-IQA error: {e}, falling back to custom implementation")
    
    # Fallback implementation
    try:
        img_array = np.array(image)
        img_float = img_as_float(img_array)
        
        if CV2_AVAILABLE:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        else:
            gray = np.mean(img_array, axis=2)
            gy, gx = np.gradient(gray)
            laplacian_var = np.sqrt(gx**2 + gy**2).var()
        
        sharpness_score = min(100, laplacian_var / 10)
        contrast = np.std(img_float) * 100
        contrast_score = min(100, contrast * 2)
        
        rg = np.abs(img_float[:,:,0] - img_float[:,:,1])
        yb = np.abs(0.5 * (img_float[:,:,0] + img_float[:,:,1]) - img_float[:,:,2])
        colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2) * 100
        color_score = min(100, colorfulness * 3)
        
        overall_score = (sharpness_score * 0.4 + contrast_score * 0.3 + color_score * 0.3)
        return overall_score
        
    except Exception as e:
        print(f"CLIP-IQA proxy calculation error: {e}")
        return 0.0
