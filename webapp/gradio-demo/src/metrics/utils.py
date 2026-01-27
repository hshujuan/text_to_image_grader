"""
Utility functions for the metrics module.
"""

import base64
from io import BytesIO

# Global model caches
_clip_model = None
_clip_preprocess = None
_vqa_model = None
_vqa_processor = None


def pil_to_base64(image, max_size=1024):
    """
    Convert PIL Image to base64 data URL for VLM APIs.
    - Normalizes color mode
    - Optionally resizes for stability and cost control
    """
    # Normalize color mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Optional resize (preserve aspect ratio)
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))

    buffered = BytesIO()
    image.save(buffered, format="PNG", optimize=True)

    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def get_clip_model():
    """
    Lazy load CLIP model (singleton pattern).
    Returns: (model, preprocess) tuple or (None, None) if unavailable
    """
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            import clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
            print(f"CLIP model loaded on {device}")
        except Exception as e:
            print(f"CLIP model loading error: {e}")
    return _clip_model, _clip_preprocess


def get_vqa_model():
    """
    Lazy load VQA model (singleton pattern).
    Returns: (model, processor) tuple or (None, None) if unavailable
    """
    global _vqa_model, _vqa_processor
    if _vqa_model is None:
        try:
            from transformers import ViltProcessor, ViltForQuestionAnswering
            _vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            _vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            print("VQA model loaded")
        except Exception as e:
            print(f"VQA model loading error: {e}")
    return _vqa_model, _vqa_processor
