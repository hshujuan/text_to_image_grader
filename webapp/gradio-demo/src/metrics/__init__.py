"""
Metrics Module for Text-to-Image Quality Assessment
====================================================

This module provides all metric calculations used by the T2I grader.

Metric Hierarchy:
-----------------
1. NORTH STAR: Soft-TIFA GM (Atomic Faithfulness)
2. SUPPORTING:
   A. Image Quality: BRISQUE, NIQE, CLIP-IQA
   B. Alignment: CLIPScore, VQAScore, AHEaD, PickScore, TIFA, DSG, PSG, VPEval
   C. Safety: T2ISafety (Toxicity, Fairness, Privacy)

Usage:
------
from metrics import (
    # North Star
    calculate_soft_tifa_score,
    
    # Image Quality
    calculate_brisque_score,
    calculate_niqe_score,
    calculate_clip_iqa_score,
    
    # Alignment
    calculate_real_clipscore,
    calculate_real_vqascore,
    calculate_ahead_score,
    calculate_pickscore_proxy,
    calculate_tifa_score,
    calculate_dsg_score,
    calculate_psg_score,
    calculate_vpeval_score,
    
    # Safety
    evaluate_t2i_safety,
    
    # Utilities
    pil_to_base64,
    get_clip_model,
    get_vqa_model,
)
"""

# North Star Metric
from .soft_tifa import calculate_soft_tifa_score

# Image Quality Metrics
from .image_quality import (
    calculate_brisque_score,
    calculate_niqe_score,
    calculate_clip_iqa_score,
)

# Alignment Metrics
from .alignment import (
    calculate_real_clipscore,
    calculate_real_vqascore,
    calculate_ahead_score,
    calculate_pickscore_proxy,
    calculate_tifa_score,
    calculate_dsg_score,
    calculate_psg_score,
    calculate_vpeval_score,
)

# Safety Metrics
from .safety import evaluate_t2i_safety

# Utilities
from .utils import (
    pil_to_base64,
    get_clip_model,
    get_vqa_model,
)

__all__ = [
    # North Star
    'calculate_soft_tifa_score',
    
    # Image Quality
    'calculate_brisque_score',
    'calculate_niqe_score',
    'calculate_clip_iqa_score',
    
    # Alignment
    'calculate_real_clipscore',
    'calculate_real_vqascore',
    'calculate_ahead_score',
    'calculate_pickscore_proxy',
    'calculate_tifa_score',
    'calculate_dsg_score',
    'calculate_psg_score',
    'calculate_vpeval_score',
    
    # Safety
    'evaluate_t2i_safety',
    
    # Utilities
    'pil_to_base64',
    'get_clip_model',
    'get_vqa_model',
]
