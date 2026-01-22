import json
import re
import numpy as np
from typing import List, Tuple
from .utils import pil_to_base64

def calculate_soft_tifa_pure_gm(image, prompt, client, model) -> Tuple[float, List[str], List[float]]:
    """
    Pure Geometric Mean Soft-TIFA.
    A single 0.0 will result in a 0.0 total score. 
    The VQA prompt is enhanced to prevent 'accidental' zeros.
    """
    # 1. Atomic Fact Extraction
    extraction_prompt = f"""
    Decompose this prompt into 5-7 atomic, visually verifiable facts: "{prompt}"
    Return ONLY JSON: {{"atoms": ["fact1", "fact2"]}}
    """
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.0
        )
        json_str = re.search(r"\{.*\}", resp.choices[0].message.content, re.DOTALL).group(0)
        atoms = list(dict.fromkeys(json.loads(json_str).get("atoms", [])))
    except:
        return 0.0, [], []

    # 2. Sophisticated Scoring Rubric
    scores = []
    img_b64 = pil_to_base64(image)
    
    for atom in atoms:
        # We define a rubric to guide the model away from harsh 0.0s
        vqa_prompt = f"""
        Evaluate the image based on this criterion: "{atom}"
        
        Assign a score based on this strict rubric:
        1.0: Perfect match.
        0.7 - 0.9: Clearly present but minor attribute mismatch (e.g., wrong shade of color).
        0.4 - 0.6: Partially present or obscured.
        0.1 - 0.3: Distant hint or highly distorted version of the object is present.
        0.0: The object/attribute is completely absent or replaced by something else.

        Respond with ONLY the numerical score.
        """
        try:
            score_resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": vqa_prompt}, 
                    {"type": "image_url", "image_url": {"url": img_b64}}
                ]}],
                temperature=0.0
            )
            val_match = re.search(r"(\d?\.\d+|\d)", score_resp.choices[0].message.content)
            val = float(val_match.group(0)) if val_match else 0.0
            scores.append(max(0.0, min(1.0, val)))
        except:
            scores.append(0.0)

    # 3. Pure Geometric Mean (No epsilon floor)
    # If any score is 0.0, np.prod will be 0.0.
    if not scores:
        return 0.0, atoms, []
    
    # Mathematical GM: (p1 * p2 * ... * pn)^(1/n)
    gm_score = np.prod(scores) ** (1 / len(scores))
    
    return float(gm_score * 100), atoms, scores


# Alias for backward compatibility with imports
calculate_soft_tifa_score = calculate_soft_tifa_pure_gm