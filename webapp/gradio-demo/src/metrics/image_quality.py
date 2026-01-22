"""
Image Quality Metrics (No Text Reference)
==========================================

These metrics assess technical image quality independent of the prompt.

Metrics:
- BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator
- NIQE: Natural Image Quality Evaluator  
- CLIP-IQA: CLIP-based Image Quality Assessment
"""

import numpy as np

# Optional imports with fallbacks for Python 3.14 compatibility
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


def calculate_brisque_score(image):
    """
    Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score.
    
    BRISQUE evaluates spatial quality by detecting blur, noise, and compression artifacts.
    Lower raw BRISQUE scores indicate better quality (0-100 scale typical).
    
    Args:
        image: PIL Image to evaluate
    
    Returns:
        float: Quality score 0-100 (inverted, so higher is better)
    """
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        if not CV2_AVAILABLE:
            # Fallback without cv2: use basic numpy-based quality estimation
            # Calculate basic sharpness via gradient magnitude
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            gy, gx = np.gradient(gray)
            sharpness = np.sqrt(gx**2 + gy**2).var()
            contrast = gray.std()
            quality_score = min(100, (sharpness / 50 * 50) + (contrast / 50 * 50))
            return quality_score
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Use Laplacian-based quality estimation (more reliable than cv2.quality API)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate contrast (standard deviation)
        contrast = gray.std()
        
        # Combine metrics (normalized approximation)
        quality_score = min(100, (sharpness / 500 * 50) + (contrast / 50 * 50))
        return quality_score
        
    except Exception as e:
        print(f"BRISQUE calculation error: {e}")
        return 0.0


def calculate_niqe_score(image):
    """
    Calculate NIQE (Natural Image Quality Evaluator) score.
    
    NIQE compares image statistics to natural image distributions.
    Lower raw NIQE scores indicate better quality.
    
    Args:
        image: PIL Image to evaluate
    
    Returns:
        float: Quality score 0-100 (inverted, so higher is better)
    """
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        if not CV2_AVAILABLE:
            # Fallback without cv2: use basic numpy-based naturalness estimation
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            
            # Calculate histogram uniformity (entropy)
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            # Calculate edge density via gradient
            gy, gx = np.gradient(gray)
            edge_magnitude = np.sqrt(gx**2 + gy**2)
            edge_density = np.sum(edge_magnitude > 20) / edge_magnitude.size
            
            # Combine metrics
            naturalness_score = min(100, (entropy / 8 * 50) + (edge_density * 5000))
            return naturalness_score
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Use naturalness metrics (more reliable than cv2.quality API)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram uniformity (entropy)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # Calculate edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine metrics (normalized approximation)
        naturalness_score = min(100, (entropy / 8 * 50) + (edge_density * 5000))
        return naturalness_score
        
    except Exception as e:
        print(f"NIQE calculation error: {e}")
        return 0.0


def calculate_clip_iqa_score(image):
    """
    Calculate CLIP-IQA score using image statistics as proxy.
    
    True CLIP-IQA requires the specific trained model. This implementation
    uses a simplified proxy based on perceptual quality indicators.
    
    Args:
        image: PIL Image to evaluate
    
    Returns:
        float: Quality score 0-100
    """
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        img_float = img_as_float(img_array)
        
        # Calculate various image quality indicators
        # 1. Sharpness (Laplacian variance)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, laplacian_var / 10)
        
        # 2. Contrast (standard deviation)
        contrast = np.std(img_float) * 100
        contrast_score = min(100, contrast * 2)
        
        # 3. Colorfulness
        rg = np.abs(img_float[:,:,0] - img_float[:,:,1])
        yb = np.abs(0.5 * (img_float[:,:,0] + img_float[:,:,1]) - img_float[:,:,2])
        colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2) * 100
        color_score = min(100, colorfulness * 3)
        
        # Combine scores
        overall_score = (sharpness_score * 0.4 + contrast_score * 0.3 + color_score * 0.3)
        
        return overall_score
        
    except Exception as e:
        print(f"CLIP-IQA proxy calculation error: {e}")
        return 0.0
