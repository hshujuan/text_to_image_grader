"""
Soft-TIFA GM: North Star Metric for Text-to-Image Quality Assessment
=====================================================================

Soft-TIFA (Soft Text-Image Faithfulness through Atomic evaluation) measures 
how well an image satisfies atomic visual criteria extracted from the prompt.

Methodology:
1. Extract 5-7 atomic visual facts from prompt
2. Score each atom probabilistically (0.0 to 1.0)
3. Calculate Geometric Mean: (score₁ × score₂ × ... × scoreₙ)^(1/n) × 100

Why Geometric Mean?
- Compositional AND logic: ALL criteria must be satisfied
- Penalizes missing elements (single 0 = overall 0)
- Better correlation with human judgment on completeness
"""

import json
import numpy as np
from .utils import pil_to_base64


def calculate_soft_tifa_score(image, prompt, client, model):
    """
    Calculate true Soft-TIFA Geometric Mean score.
    
    This is the NORTH STAR METRIC for text-to-image evaluation.
    
    Args:
        image: PIL Image to evaluate
        prompt: Original text prompt
        client: Azure OpenAI client instance
        model: Model deployment name (e.g., "gpt-4o")
    
    Returns:
        tuple: (gm_score, atoms_list, individual_scores)
            - gm_score: Geometric mean score (0-100)
            - atoms_list: List of extracted atomic criteria
            - individual_scores: List of scores for each atom (0.0-1.0)
    """
    # Stage 1: Extract atomic visual criteria
    extraction_prompt = f"""
Analyze this text-to-image prompt: "{prompt}"

Extract 5-7 atomic visual facts that MUST be present in the generated image.
Focus on: objects, attributes (colors, styles), spatial relationships, and composition.

Return ONLY a valid JSON object:
{{"atoms": ["fact 1", "fact 2", "fact 3", ...]}}
"""
    
    try:
        atoms_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a prompt analysis expert. Extract atomic visual criteria."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        if not atoms_response.choices or len(atoms_response.choices) == 0:
            print("No response received from atom extraction")
            return 0.0, [], []
        
        atoms_content = atoms_response.choices[0].message.content.strip()
        atoms_content = atoms_content.replace('```json', '').replace('```', '').strip()
        atoms = json.loads(atoms_content)['atoms']
    except Exception as e:
        print(f"Soft-TIFA atom extraction error: {e}")
        return 0.0, [], []
    
    # Stage 2: Score each atom individually
    img_base64 = pil_to_base64(image)
    criteria_scores = []
    
    for atom in atoms:
        vqa_prompt = f"""
Look at this image and evaluate if the following criterion is met:
Criterion: "{atom}"

Provide a probability score from 0.0 to 1.0:
- 1.0 = Criterion fully met
- 0.5 = Partially met
- 0.0 = Not met at all

Respond with ONLY a number between 0.0 and 1.0.
"""
        
        try:
            prob_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise image evaluator. Respond with only a probability score."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vqa_prompt},
                            {"type": "image_url", "image_url": {"url": img_base64}}
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=10
            )
            if prob_response.choices and len(prob_response.choices) > 0:
                prob = float(prob_response.choices[0].message.content.strip())
                prob = max(0.0, min(1.0, prob))
                criteria_scores.append(prob)
            else:
                criteria_scores.append(0.0)
        except Exception as e:
            print(f"Error scoring criterion '{atom}': {e}")
            criteria_scores.append(0.0)
    
    # Stage 3: Calculate Geometric Mean
    if criteria_scores:
        gm_score = (np.prod(criteria_scores)) ** (1 / len(criteria_scores))
    else:
        gm_score = 0.0
    
    return gm_score * 100, atoms, criteria_scores
