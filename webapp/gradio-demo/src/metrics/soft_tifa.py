# Copyright: Meta Platforms, Inc. and affiliates
# Adapted from GenEval2: https://github.com/facebookresearch/GenEval2
# License: CC BY-NC 4.0

"""
Soft-TIFA Metric Implementation
===============================

Based on the original GenEval2 implementation by Meta FAIR.
Reference: "GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation"
Paper: https://arxiv.org/abs/2512.16853

Soft-TIFA uses a VQA model to query the generated image with each of the 
associated list of questions. It assigns a soft score to each question based 
on the VQA model's probability assigned to the correct answer when given the image.

- Soft-TIFA AM (Arithmetic Mean): Atom-level estimate of model performance
- Soft-TIFA GM (Geometric Mean): Prompt-level estimate of model performance

This implementation:
1. First checks GenEval2 benchmark data for pre-defined VQA pairs (if prompt matches)
2. Falls back to LLM-generated VQA pairs for custom prompts
3. Uses Azure OpenAI (GPT-4o) as the VQA model backend
"""

import json
import re
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from scipy.stats import gmean
from .utils import pil_to_base64


# ============================================================================
# GenEval2 Benchmark Data Loading
# ============================================================================

_GENEVAL2_DATA: Optional[Dict[str, dict]] = None

def _load_geneval2_data() -> Dict[str, dict]:
    """
    Load GenEval2 benchmark data with pre-defined VQA pairs.
    Uses lazy loading and caching for efficiency.
    
    Returns:
        Dictionary mapping prompts to their VQA data
    """
    global _GENEVAL2_DATA
    
    if _GENEVAL2_DATA is not None:
        return _GENEVAL2_DATA
    
    _GENEVAL2_DATA = {}
    
    # Try to find geneval2_data.jsonl in lib/geneval2/
    possible_paths = [
        Path(__file__).parent.parent.parent.parent.parent / "lib" / "geneval2" / "geneval2_data.jsonl",
        Path(__file__).parent.parent.parent.parent.parent.parent / "lib" / "geneval2" / "geneval2_data.jsonl",
        Path("lib/geneval2/geneval2_data.jsonl"),
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            prompt = entry.get("prompt", "").strip().lower()
                            if prompt:
                                _GENEVAL2_DATA[prompt] = entry
                print(f"Loaded {len(_GENEVAL2_DATA)} GenEval2 benchmark prompts from {data_path}")
                break
            except Exception as e:
                print(f"Failed to load GenEval2 data from {data_path}: {e}")
    
    if not _GENEVAL2_DATA:
        print("GenEval2 benchmark data not found. Using LLM-generated VQA pairs only.")
    
    return _GENEVAL2_DATA


def _get_predefined_vqa_list(prompt: str) -> Optional[List[Tuple[str, str]]]:
    """
    Get pre-defined VQA pairs from GenEval2 benchmark if prompt matches.
    
    Args:
        prompt: The text prompt to look up
        
    Returns:
        List of (question, answer) tuples if found, None otherwise
    """
    data = _load_geneval2_data()
    normalized_prompt = prompt.strip().lower()
    
    if normalized_prompt in data:
        entry = data[normalized_prompt]
        vqa_list = entry.get("vqa_list", [])
        return [(q, a) for q, a in vqa_list]
    
    return None


# ============================================================================
# Rule-Based VQA Generation (following GenEval2 methodology)
# ============================================================================

# Count words that GenEval2 uses
COUNT_WORDS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

# Common colors
COLORS = [
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 
    'black', 'white', 'gray', 'grey', 'golden', 'silver', 'beige', 'cyan',
    'magenta', 'turquoise', 'maroon', 'navy', 'teal', 'olive', 'coral'
]

# Common materials
MATERIALS = [
    'wooden', 'metal', 'plastic', 'glass', 'leather', 'fabric', 'stone',
    'concrete', 'rubber', 'paper', 'cardboard', 'ceramic', 'steel', 'iron',
    'bronze', 'copper', 'gold', 'silver', 'aluminum', 'marble', 'granite'
]

# Common sizes/shapes
SIZES = ['small', 'large', 'big', 'tiny', 'huge', 'giant', 'little', 'tall', 'short', 'long']

# Common textures/states
TEXTURES = [
    'fluffy', 'smooth', 'rough', 'shiny', 'matte', 'glossy', 'fuzzy',
    'soft', 'hard', 'wet', 'dry', 'old', 'new', 'ancient', 'modern'
]

# Spatial relations
SPATIAL_RELATIONS = [
    'in front of', 'behind', 'next to', 'beside', 'on top of', 'under',
    'below', 'above', 'to the left of', 'to the right of', 'near',
    'between', 'inside', 'outside', 'on', 'over', 'beneath', 'around'
]

# Action verbs (present participle forms)
ACTION_VERBS = [
    'running', 'walking', 'sitting', 'standing', 'flying', 'swimming',
    'jumping', 'sleeping', 'eating', 'drinking', 'playing', 'reading',
    'writing', 'dancing', 'singing', 'climbing', 'falling', 'fighting',
    'laughing', 'crying', 'smiling', 'talking', 'looking', 'watching'
]

# All attributes combined
ALL_ATTRIBUTES = COLORS + MATERIALS + SIZES + TEXTURES


def _extract_vqa_list_rulebased(prompt: str) -> List[Tuple[str, str]]:
    """
    Rule-based VQA extraction following GenEval2 methodology.
    
    Parses the prompt to identify:
    1. Objects (nouns)
    2. Counts (number words)
    3. Attributes (colors, materials, sizes)
    4. Spatial relations
    5. Actions (verbs)
    
    Returns a list of (question, answer, skill) tuples.
    """
    prompt_lower = prompt.lower().strip()
    words = prompt_lower.split()
    vqa_list = []
    skills = []
    
    # Track found objects and their attributes
    found_objects = []
    object_attributes = {}  # object -> list of attributes
    object_counts = {}  # object -> count word
    
    # Step 1: Find counts and associate with following noun
    i = 0
    while i < len(words):
        word = words[i].strip('.,!?')
        
        # Check for count word
        if word in COUNT_WORDS:
            count = word
            # Look for attributes and object after count
            attrs = []
            obj = None
            j = i + 1
            while j < len(words):
                next_word = words[j].strip('.,!?')
                if next_word in ALL_ATTRIBUTES:
                    attrs.append(next_word)
                    j += 1
                elif next_word in SPATIAL_RELATIONS or next_word == 'and':
                    break
                elif next_word not in ['a', 'an', 'the', 'of']:
                    # This is likely the object
                    obj = next_word
                    # Check if it's plural, if so, keep it
                    if j + 1 < len(words) and words[j + 1].strip('.,!?') not in SPATIAL_RELATIONS + ['and']:
                        # Might be a compound noun
                        pass
                    break
                else:
                    j += 1
            
            if obj:
                found_objects.append(obj)
                object_counts[obj] = count
                object_attributes[obj] = attrs
            i = j + 1 if obj else i + 1
        else:
            i += 1
    
    # Step 2: Find objects without explicit counts (implied "a" or "one")
    # Look for pattern: [attribute] [object]
    for i, word in enumerate(words):
        word = word.strip('.,!?')
        if word in ALL_ATTRIBUTES:
            # Look for object after attribute
            for j in range(i + 1, min(i + 3, len(words))):
                next_word = words[j].strip('.,!?')
                if next_word not in ALL_ATTRIBUTES + ['a', 'an', 'the', 'and', 'of'] + SPATIAL_RELATIONS:
                    if next_word not in found_objects and next_word not in COUNT_WORDS:
                        found_objects.append(next_word)
                        object_attributes.setdefault(next_word, []).append(word)
                        if next_word not in object_counts:
                            object_counts[next_word] = 'one'  # Implied single
                    break
    
    # Step 3: Find standalone objects (nouns without attributes)
    for i, word in enumerate(words):
        word = word.strip('.,!?')
        if word == 'a' or word == 'an':
            if i + 1 < len(words):
                next_word = words[i + 1].strip('.,!?')
                if next_word not in found_objects and next_word not in COUNT_WORDS + ALL_ATTRIBUTES:
                    found_objects.append(next_word)
                    object_counts[next_word] = 'one'
    
    # Step 4: Find spatial relations
    spatial_pairs = []
    for relation in SPATIAL_RELATIONS:
        if relation in prompt_lower:
            # Find what's before and after the relation
            parts = prompt_lower.split(relation)
            if len(parts) >= 2:
                spatial_pairs.append((relation, parts[0].strip(), parts[1].strip()))
    
    # Step 5: Find actions
    found_actions = []
    for verb in ACTION_VERBS:
        if verb in prompt_lower:
            found_actions.append(verb)
    
    # =========================================================================
    # Generate VQA pairs following GenEval2 template
    # =========================================================================
    
    for obj in found_objects:
        # Pluralize for questions if count > 1
        obj_plural = obj if obj.endswith('s') else obj + 's'
        obj_singular = obj.rstrip('s') if obj.endswith('s') else obj
        
        count = object_counts.get(obj, 'one')
        attrs = object_attributes.get(obj, [])
        
        # 1. Count question
        vqa_list.append((f"How many {obj_plural} are in the image?", count))
        skills.append("count")
        
        # 2. Attribute questions
        for attr in attrs:
            if count == 'one':
                vqa_list.append((f"Is the {obj_singular} {attr}?", "Yes"))
            else:
                vqa_list.append((f"Are the {obj_plural} {attr}?", "Yes"))
            skills.append("attribute")
        
        # 3. Object presence question
        vqa_list.append((f"Are there any {obj_plural} in the image?", "Yes"))
        skills.append("object")
    
    # 4. Spatial relation questions
    for relation, before, after in spatial_pairs:
        # Try to extract the objects involved
        # This is simplified - real implementation would need NLP
        vqa_list.append((f"Is there something {relation} something else?", "Yes"))
        skills.append("position")
    
    # 5. Action questions
    for verb in found_actions:
        vqa_list.append((f"Is something {verb} in the image?", "Yes"))
        skills.append("verb")
    
    return vqa_list, skills


def _return_numeric_string(number: str) -> str:
    """Convert word numbers to digit strings (from GenEval2)."""
    mapping = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    return mapping.get(number.lower(), 'other')


def _extract_vqa_list_llm(prompt: str, client, model) -> List[Tuple[str, str]]:
    """
    Extract VQA question-answer pairs from a prompt.
    
    Following GenEval2 methodology:
    - Questions for counting: "How many X are in the image?"
    - Questions for presence: "Are there any X in the image?"
    - Questions for attributes: "Is/Are the X [attribute]?"
    - Questions for relations: "Is/Are X [relation] Y?"
    
    Returns a list of (question, answer) tuples.
    """
    extraction_prompt = f"""Analyze this text-to-image prompt and generate VQA question-answer pairs for evaluation.

Prompt: "{prompt}"

Generate questions to verify each visual element:
1. For objects: "Are there any [object] in the image?" → "Yes"
2. For counting: "How many [object] are in the image?" → "[count word]" (e.g., "one", "two", "three")
3. For attributes (color, material, size, etc.): "Is/Are the [object] [attribute]?" → "Yes"
4. For spatial relations: "Is/Are the [object1] [relation] the [object2]?" → "Yes"
5. For actions/verbs: "Is the [subject] [verb]-ing?" → "Yes"

Return ONLY a JSON array with question-answer pairs:
{{"vqa_list": [["question1", "answer1"], ["question2", "answer2"], ...]}}

Important:
- For Yes/No questions, answer should be "Yes" 
- For counting questions, answer should be the count word (e.g., "two", "three")
- Cover all visual elements mentioned in the prompt
- Keep questions simple and unambiguous"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.0
        )
        json_match = re.search(r"\{.*\}", resp.choices[0].message.content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            vqa_list = data.get("vqa_list", [])
            # Validate and convert to list of tuples
            return [(q, a) for q, a in vqa_list if len([q, a]) == 2]
    except Exception as e:
        print(f"VQA extraction failed: {e}")
    
    return []


def _vqa_score_single(question: str, answer: str, img_b64: str, client, model) -> float:
    """
    Query VQA model and get soft score for a single question.
    
    Following GenEval2 methodology:
    - For counting questions: Check if the answer word or its numeric equivalent is present
    - For Yes/No questions: Check if "Yes" is the response
    
    Returns a soft score between 0.0 and 1.0 based on model confidence.
    """
    # Construct answer list based on question type (following GenEval2)
    if question.lower().startswith("how many"):
        # For counting questions, accept both word and numeric forms
        answer_variants = [
            answer, answer.capitalize(), ' ' + answer, ' ' + answer.capitalize(),
            _return_numeric_string(answer), ' ' + _return_numeric_string(answer)
        ]
    else:
        # For Yes/No questions
        answer_variants = ['Yes', 'yes', ' yes', ' Yes']
    
    vqa_prompt = f"""{question} Answer in one word."""
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_b64}},
                {"type": "text", "text": vqa_prompt}
            ]}],
            temperature=0.0,
            max_tokens=10,
            logprobs=True,  # Request log probabilities if supported
            top_logprobs=5
        )
        
        response_text = resp.choices[0].message.content.strip()
        
        # Try to extract probability from logprobs if available
        if hasattr(resp.choices[0], 'logprobs') and resp.choices[0].logprobs:
            logprobs = resp.choices[0].logprobs
            if hasattr(logprobs, 'content') and logprobs.content:
                # Sum probabilities for all answer variants
                import math
                total_prob = 0.0
                for token_info in logprobs.content:
                    if hasattr(token_info, 'top_logprobs'):
                        for top_lp in token_info.top_logprobs:
                            if any(av.lower() in top_lp.token.lower() for av in answer_variants):
                                total_prob = max(total_prob, math.exp(top_lp.logprob))
                if total_prob > 0:
                    return min(1.0, total_prob)
        
        # Fallback: Binary check with confidence estimation
        response_lower = response_text.lower().strip()
        
        # Check for exact or partial match
        for av in answer_variants:
            if av.lower().strip() in response_lower or response_lower in av.lower().strip():
                return 1.0  # Strong match
        
        # For counting questions, check numeric equivalence
        if question.lower().startswith("how many"):
            try:
                # Try to extract a number from the response
                num_match = re.search(r'\d+', response_text)
                expected_num = _return_numeric_string(answer)
                if num_match and expected_num != 'other':
                    if num_match.group(0) == expected_num:
                        return 1.0
                    # Partial credit for close counts
                    diff = abs(int(num_match.group(0)) - int(expected_num))
                    if diff == 1:
                        return 0.5
                    elif diff == 2:
                        return 0.25
            except:
                pass
        
        # Check for negative responses
        if any(neg in response_lower for neg in ['no', 'not', 'none', 'zero', '0']):
            return 0.0
            
        return 0.0
        
    except Exception as e:
        print(f"VQA query failed: {e}")
        return 0.0


def soft_tifa(image, prompt: str, client, model, method: str = "gm") -> Tuple[float, List[str], List[float]]:
    """
    Soft-TIFA evaluation following GenEval2 methodology.
    
    VQA pair sources (in order of priority):
    1. GenEval2 benchmark pre-defined pairs (800 prompts)
    2. Rule-based extraction (following GenEval2 patterns)
    3. LLM-generated pairs (fallback for complex prompts)
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt used to generate the image
        client: API client for VQA model
        model: Model deployment name
        method: "gm" for geometric mean (prompt-level), "am" for arithmetic mean (atom-level)
    
    Returns:
        Tuple of (score, questions, per_question_scores)
        - score: 0-100 scaled score
        - questions: List of VQA questions used
        - per_question_scores: List of scores for each question (0.0-1.0)
    """
    vqa_list = None
    vqa_source = None
    
    # Priority 1: Try GenEval2 benchmark pre-defined pairs
    vqa_list = _get_predefined_vqa_list(prompt)
    if vqa_list:
        vqa_source = "GenEval2 benchmark"
    
    # Priority 2: Try rule-based extraction
    if not vqa_list:
        vqa_list, _ = _extract_vqa_list_rulebased(prompt)
        if vqa_list:
            vqa_source = "rule-based"
    
    # Priority 3: Fall back to LLM-generated pairs
    if not vqa_list:
        vqa_list = _extract_vqa_list_llm(prompt, client, model)
        if vqa_list:
            vqa_source = "LLM-generated"
    
    if not vqa_list:
        print("Warning: No VQA pairs could be generated for prompt")
        return 0.0, [], []
    
    print(f"Using {len(vqa_list)} VQA pairs from {vqa_source}")
    
    # Convert image to base64
    img_b64 = pil_to_base64(image)
    
    # Score each VQA pair
    score_list = []
    questions = []
    
    for question, answer in vqa_list:
        questions.append(f"{question} → {answer}")
        ans_prob = _vqa_score_single(question, answer, img_b64, client, model)
        score_list.append(ans_prob)
    
    if not score_list:
        return 0.0, questions, []
    
    # Aggregate scores
    if method == "gm":
        # Geometric Mean (prompt-level) - following GenEval2
        # Add small epsilon to avoid zero multiplication issues
        epsilon = 1e-10
        adjusted_scores = [max(s, epsilon) for s in score_list]
        final_score = float(gmean(adjusted_scores))
    else:
        # Arithmetic Mean (atom-level)
        final_score = sum(score_list) / len(score_list)
    
    return float(final_score * 100), questions, score_list


def calculate_soft_tifa_gm(image, prompt: str, client, model) -> Tuple[float, List[str], List[float]]:
    """
    Soft-TIFA with Geometric Mean aggregation.
    
    Best for prompt-level evaluation where all atoms should be present.
    A single failed atom significantly reduces the score.
    
    Returns:
        Tuple of (score, questions, per_question_scores)
    """
    return soft_tifa(image, prompt, client, model, method="gm")


def calculate_soft_tifa_am(image, prompt: str, client, model) -> Tuple[float, List[str], List[float]]:
    """
    Soft-TIFA with Arithmetic Mean aggregation.
    
    Best for atom-level evaluation.
    Treats each atom equally regardless of other atoms.
    
    Returns:
        Tuple of (score, questions, per_question_scores)
    """
    return soft_tifa(image, prompt, client, model, method="am")


# Default export: GM for prompt-level evaluation (following GenEval2 paper recommendation)
calculate_soft_tifa_score = calculate_soft_tifa_gm
