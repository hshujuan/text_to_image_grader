import json
import re
import numpy as np
import os
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import pil_to_base64

# Environment variable to control batch mode
USE_BATCH_SOFT_TIFA = os.environ.get("USE_BATCH_SOFT_TIFA", "true").lower() == "true"


def _verify_single_atom(atom: str, img_b64: str, client, model) -> float:
    """Verify a single atom against the image. Used for parallel execution."""
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
        return max(0.0, min(1.0, val))
    except:
        return 0.0


def calculate_soft_tifa_pure_gm(image, prompt, client, model) -> Tuple[float, List[str], List[float]]:
    """
    Pure Geometric Mean Soft-TIFA.
    A single 0.0 will result in a 0.0 total score. 
    
    Performance optimizations:
    - USE_BATCH_SOFT_TIFA=true (default): Verifies all atoms in a SINGLE API call
    - USE_BATCH_SOFT_TIFA=false: Parallel verification calls (faster but uses more API quota)
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

    if not atoms:
        return 0.0, [], []

    img_b64 = pil_to_base64(image)
    
    # 2. Score atoms - BATCHED mode (single API call, much faster!)
    if USE_BATCH_SOFT_TIFA:
        scores = _verify_atoms_batch(atoms, img_b64, client, model)
    else:
        # PARALLEL mode - multiple concurrent API calls
        scores = _verify_atoms_parallel(atoms, img_b64, client, model)

    # 3. Pure Geometric Mean (No epsilon floor)
    if not scores:
        return 0.0, atoms, []
    
    gm_score = np.prod(scores) ** (1 / len(scores))
    
    return float(gm_score * 100), atoms, scores


def _verify_atoms_batch(atoms: List[str], img_b64: str, client, model) -> List[float]:
    """
    Verify ALL atoms in a SINGLE API call (much faster!).
    Returns scores in the same order as atoms.
    """
    atoms_list = "\n".join([f"{i+1}. {atom}" for i, atom in enumerate(atoms)])
    
    batch_prompt = f"""
    Evaluate the image for EACH of these criteria and assign a score:
    
{atoms_list}

    Scoring rubric for EACH criterion:
    1.0: Perfect match.
    0.7 - 0.9: Clearly present but minor attribute mismatch.
    0.4 - 0.6: Partially present or obscured.
    0.1 - 0.3: Distant hint or highly distorted version.
    0.0: Completely absent or replaced by something else.

    Respond with ONLY a JSON object mapping each criterion number to its score.
    Example: {{"1": 0.9, "2": 0.7, "3": 1.0, "4": 0.5, "5": 0.8}}
    """
    
    try:
        score_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": batch_prompt}, 
                {"type": "image_url", "image_url": {"url": img_b64}}
            ]}],
            temperature=0.0
        )
        
        # Parse JSON response
        response_text = score_resp.choices[0].message.content
        json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
        if json_match:
            scores_dict = json.loads(json_match.group(0))
            scores = []
            for i in range(len(atoms)):
                key = str(i + 1)
                val = float(scores_dict.get(key, 0.0))
                scores.append(max(0.0, min(1.0, val)))
            return scores
        
        # Fallback: try to extract numbers in order
        numbers = re.findall(r"(\d?\.\d+|\d)", response_text)
        scores = [max(0.0, min(1.0, float(n))) for n in numbers[:len(atoms)]]
        while len(scores) < len(atoms):
            scores.append(0.0)
        return scores
        
    except Exception as e:
        print(f"Batch verification failed: {e}, falling back to parallel")
        return _verify_atoms_parallel(atoms, img_b64, client, model)


def _verify_atoms_parallel(atoms: List[str], img_b64: str, client, model) -> List[float]:
    """
    Verify atoms in PARALLEL using ThreadPoolExecutor.
    Faster than sequential, but uses more API quota.
    """
    scores = [0.0] * len(atoms)
    
    with ThreadPoolExecutor(max_workers=min(len(atoms), 4)) as executor:
        future_to_idx = {
            executor.submit(_verify_single_atom, atom, img_b64, client, model): i
            for i, atom in enumerate(atoms)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                scores[idx] = future.result()
            except:
                scores[idx] = 0.0
    
    return scores


# Alias for backward compatibility with imports
calculate_soft_tifa_score = calculate_soft_tifa_pure_gm