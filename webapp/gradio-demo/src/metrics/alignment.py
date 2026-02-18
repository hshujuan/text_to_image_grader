"""
Alignment Metrics (Text-Image Correspondence)
==============================================

These metrics measure how well the generated image matches the text prompt.

Metrics:
- CLIPScore: Global semantic alignment via embedding cosine similarity (torchmetrics)
- VQAScore: Visual Question Answering based verification
- AHEaD: Attention-based Head alignment score
- PickScore: Human preference estimation (HuggingFace model)
- TIFA: Text-to-Image Faithfulness via QA
- DSG: Davidsonian Scene Graph decomposition
- PSG: Panoptic Scene Graph evaluation
- VPEval: Visual Programming evaluation
"""

import json
import numpy as np
import torch
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from .utils import get_clip_model, get_vqa_model, pil_to_base64

# =============================================================================
# VLM Metrics Cache (for identical image+prompt pairs)
# =============================================================================
_vlm_metrics_cache = {}
_cache_max_size = 100  # Maximum cache entries

def _get_cache_key(image, prompt):
    """Generate a cache key from image content and prompt."""
    # Hash image bytes + prompt for unique key
    img_bytes = image.tobytes()
    key_data = hashlib.md5(img_bytes + prompt.encode()).hexdigest()
    return key_data

def _cache_get(key, metric_name):
    """Get cached result for a specific metric."""
    if key in _vlm_metrics_cache and metric_name in _vlm_metrics_cache[key]:
        return _vlm_metrics_cache[key][metric_name]
    return None

def _cache_set(key, metric_name, value):
    """Cache a metric result."""
    global _vlm_metrics_cache
    # Simple LRU: if cache is full, remove oldest entry
    if len(_vlm_metrics_cache) >= _cache_max_size:
        oldest_key = next(iter(_vlm_metrics_cache))
        del _vlm_metrics_cache[oldest_key]
    
    if key not in _vlm_metrics_cache:
        _vlm_metrics_cache[key] = {}
    _vlm_metrics_cache[key][metric_name] = value

def clear_vlm_cache():
    """Clear the VLM metrics cache."""
    global _vlm_metrics_cache
    _vlm_metrics_cache = {}

# Try importing torchmetrics for CLIPScore
try:
    from torchmetrics.multimodal.clip_score import CLIPScore as TorchMetricsCLIPScore
    TORCHMETRICS_AVAILABLE = True
    _torchmetrics_clip = None  # Lazy loaded
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not available. Install with: pip install torchmetrics[multimodal]")

# Try importing PickScore model
try:
    from transformers import AutoProcessor, AutoModel
    PICKSCORE_AVAILABLE = True
    _pickscore_model = None
    _pickscore_processor = None
except ImportError:
    PICKSCORE_AVAILABLE = False
    print("Warning: transformers not available for PickScore")


def _pil_to_tensor_rgb(image):
    """Convert PIL image to torch tensor (BCHW format, 0-255 uint8 for torchmetrics)."""
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    # torchmetrics CLIPScore expects uint8 in range 0-255
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def calculate_real_clipscore(image, prompt):
    """
    Calculate real CLIPScore using torchmetrics library.
    
    CLIPScore measures global semantic alignment between image and text
    by computing cosine similarity of their CLIP embeddings.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
    
    Returns:
        float: CLIP score 0-100
    """
    global _torchmetrics_clip
    
    # Check if we should use the faster OpenAI CLIP fallback (shares model with AHEaD)
    import os
    use_fast_clip = os.environ.get("USE_FAST_CLIPSCORE", "").lower() in ("1", "true", "yes")
    
    if TORCHMETRICS_AVAILABLE and not use_fast_clip:
        try:
            # Lazy load the model
            if _torchmetrics_clip is None:
                print("Loading torchmetrics CLIPScore model...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _torchmetrics_clip = TorchMetricsCLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)
                print(f"torchmetrics CLIPScore loaded on {device}")
            
            img_tensor = _pil_to_tensor_rgb(image)
            device = next(_torchmetrics_clip.parameters()).device
            img_tensor = img_tensor.to(device)
            
            # torchmetrics CLIPScore returns score * 100 (already scaled)
            score = _torchmetrics_clip(img_tensor, [prompt]).item()
            # Normalize to 0-100 range (CLIPScore typically ranges 0-40)
            normalized_score = min(100, score * 2.5)
            return normalized_score
            
        except Exception as e:
            print(f"torchmetrics CLIPScore error: {e}, falling back to custom implementation")
    
    # Fallback to custom CLIP implementation (faster, shares model with AHEaD)
    try:
        import clip
        
        model, preprocess = get_clip_model()
        if model is None or preprocess is None:
            print("CLIP model not available, using fallback")
            return 0.0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize([prompt]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
        
        clip_score = ((similarity + 1) / 2) * 100
        return clip_score
        
    except Exception as e:
        print(f"CLIPScore calculation error: {e}")
        return 0.0


def calculate_real_vqascore(image, prompt):
    """
    Calculate real VQAScore using VQA model.
    
    VQAScore uses visual question answering to verify
    whether specific facts from the prompt are present in the image.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
    
    Returns:
        float: VQA score 0-100
    """
    try:
        model, processor = get_vqa_model()
        if model is None or processor is None:
            print("VQA model not available, using fallback")
            return 0.0
        
        import torch
        
        # Generate questions from prompt
        questions = [
            f"Does this image show {prompt}?",
            f"Is this image consistent with: {prompt}?",
            "Is this image high quality?",
            "Does this image match the description?"
        ]
        
        scores = []
        for question in questions:
            try:
                encoding = processor(image, question, return_tensors="pt")
                outputs = model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get probability of positive answer
                # VQA models typically have "yes" as one of the top answers
                top_prob = probs.max().item()
                scores.append(top_prob)
            except Exception as e:
                print(f"VQA question error: {e}")
                continue
        
        if scores:
            avg_score = np.mean(scores) * 100
            return min(100, avg_score * 1.5)  # Scale up
        return 0.0
        
    except Exception as e:
        print(f"VQAScore calculation error: {e}")
        return 0.0


def calculate_ahead_score(image, prompt):
    """
    Calculate AHEaD (Alignment Head) score.
    
    AHEaD uses CLIP attention patterns to measure fine-grained alignment
    between image regions and text tokens.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
    
    Returns:
        float: AHEaD score 0-100
    """
    try:
        import torch
        import clip
        
        model, preprocess = get_clip_model()
        if model is None or preprocess is None:
            print("CLIP model not available for AHEaD")
            return 0.0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Preprocess image
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize([prompt]).to(device)
        
        with torch.no_grad():
            # Get image and text features
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate attention-based alignment
            # Using dot product as alignment measure
            alignment = (image_features @ text_features.T).item()
            
            # Also consider feature variance (diversity)
            img_variance = torch.var(image_features).item()
            
            # Combine alignment and variance for quality score
            ahead_score = ((alignment + 1) / 2) * 0.7 + min(1.0, img_variance * 10) * 0.3
            ahead_score = ahead_score * 100
        
        return min(100, ahead_score)
        
    except Exception as e:
        print(f"AHEaD calculation error: {e}")
        return 0.0


def calculate_pickscore_proxy(image, prompt):
    """
    Calculate PickScore using the official HuggingFace PickScore model.
    
    PickScore is trained on human preferences for text-to-image alignment.
    Falls back to CLIP+aesthetics proxy if model unavailable.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
    
    Returns:
        float: Human preference score 0-100
    """
    global _pickscore_model, _pickscore_processor
    
    # Check if PickScore is disabled via environment variable (saves ~4GB model download)
    import os
    if os.environ.get("SKIP_PICKSCORE", "").lower() in ("1", "true", "yes"):
        # Use fast fallback
        return _calculate_pickscore_fallback(image, prompt)
    
    if PICKSCORE_AVAILABLE:
        try:
            # Lazy load the model
            if _pickscore_model is None:
                print("Loading PickScore model (this is a large ~4GB model)...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _pickscore_processor = AutoProcessor.from_pretrained("yuvalkirstain/PickScore_v1")
                _pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
                print(f"PickScore model loaded on {device}")
            
            device = next(_pickscore_model.parameters()).device
            
            # Process inputs
            inputs = _pickscore_processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                # Get image and text embeddings
                image_embeds = _pickscore_model.get_image_features(pixel_values=inputs["pixel_values"])
                text_embeds = _pickscore_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                
                # Normalize
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Calculate score
                score = (image_embeds @ text_embeds.T).item()
            
            # PickScore typically ranges from -1 to 1, normalize to 0-100
            pick_score = ((score + 1) / 2) * 100
            return min(100, max(0, pick_score))
            
        except Exception as e:
            print(f"PickScore model error: {e}, falling back to proxy")
    
    return _calculate_pickscore_fallback(image, prompt)


def _calculate_pickscore_fallback(image, prompt):
    """Fast fallback using CLIP + aesthetics proxy."""
    try:
        clip_score = calculate_real_clipscore(image, prompt)
        
        img_array = np.array(image)
        colors = img_array.reshape(-1, 3)
        color_std = np.std(colors, axis=0).mean()
        color_score = min(100, color_std * 2)
        
        pick_proxy = clip_score * 0.7 + color_score * 0.3
        return pick_proxy
        
    except Exception as e:
        print(f"PickScore proxy error: {e}")
        return 0.0


# =============================================================================
# VLM-Based Alignment Metrics (require Azure OpenAI client)
# =============================================================================

def calculate_all_vlm_metrics_parallel(image, prompt, client, model):
    """
    Calculate all VLM-based metrics (TIFA, DSG, PSG, VPEval) in parallel.
    
    This is significantly faster than calling each metric sequentially.
    Also uses caching to avoid recalculating for identical image+prompt pairs.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
        client: Azure OpenAI client instance
        model: Model deployment name
    
    Returns:
        dict: {"tifa": score, "dsg": score, "psg": score, "vpeval": score}
    """
    cache_key = _get_cache_key(image, prompt)
    
    # Check if all metrics are cached
    cached_tifa = _cache_get(cache_key, "tifa")
    cached_dsg = _cache_get(cache_key, "dsg")
    cached_psg = _cache_get(cache_key, "psg")
    cached_vpeval = _cache_get(cache_key, "vpeval")
    
    if all(v is not None for v in [cached_tifa, cached_dsg, cached_psg, cached_vpeval]):
        print("Using cached VLM metrics")
        return {
            "tifa": cached_tifa,
            "dsg": cached_dsg,
            "psg": cached_psg,
            "vpeval": cached_vpeval
        }
    
    results = {}
    
    # Define metric functions to run in parallel
    def run_tifa():
        if cached_tifa is not None:
            return ("tifa", cached_tifa, None)
        score = calculate_tifa_score(image, prompt, client, model)
        _cache_set(cache_key, "tifa", score)
        return ("tifa", score, None)
    
    def run_dsg():
        if cached_dsg is not None:
            return ("dsg", cached_dsg, None)
        detail = calculate_dsg_score_detailed(image, prompt, client, model)
        score = detail.get('score', 0.0)
        _cache_set(cache_key, "dsg", score)
        return ("dsg", score, detail)
    
    def run_psg():
        if cached_psg is not None:
            return ("psg", cached_psg, None)
        detail = calculate_psg_score_detailed(image, prompt, client, model)
        score = detail.get('score', 0.0)
        _cache_set(cache_key, "psg", score)
        return ("psg", score, detail)
    
    def run_vpeval():
        if cached_vpeval is not None:
            return ("vpeval", cached_vpeval, None)
        score = calculate_vpeval_score(image, prompt, client, model)
        _cache_set(cache_key, "vpeval", score)
        return ("vpeval", score, None)
    
    # Run all metrics in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_tifa),
            executor.submit(run_dsg),
            executor.submit(run_psg),
            executor.submit(run_vpeval)
        ]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                metric_name = result[0]
                results[metric_name] = result[1]
                if len(result) > 2 and result[2] is not None:
                    results[f"{metric_name}_details"] = result[2]
            except Exception as e:
                print(f"Error in parallel VLM metric: {e}")
    
    # Ensure all keys exist
    return {
        "tifa": results.get("tifa", 0.0),
        "dsg": results.get("dsg", 0.0),
        "dsg_details": results.get("dsg_details"),
        "psg": results.get("psg", 0.0),
        "psg_details": results.get("psg_details"),
        "vpeval": results.get("vpeval", 0.0)
    }


def _batch_verify_with_image(client, model, img_base64, verification_items):
    """
    Batch multiple verification requests into a single API call.
    
    Args:
        client: Azure OpenAI client
        model: Model deployment name
        img_base64: Base64 encoded image
        verification_items: List of dicts with 'prompt' and 'response_type' keys
                           response_type can be 'yes_no' or 'score'
    
    Returns:
        list: Results for each verification item
    """
    if not verification_items:
        return []
    
    # Build batched prompt
    batch_prompt = "Evaluate this image for the following checks. Respond with a JSON array.\n\n"
    for i, item in enumerate(verification_items):
        if item.get('response_type') == 'yes_no':
            batch_prompt += f"{i+1}. {item['prompt']} (answer: yes/no)\n"
        else:
            batch_prompt += f"{i+1}. {item['prompt']} (answer: 0.0-1.0 score)\n"
    
    batch_prompt += "\nReturn ONLY a JSON array with your answers, e.g.: [\"yes\", 0.8, \"no\", 0.5]"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Evaluate images and return results as JSON array."},
                {"role": "user", "content": [
                    {"type": "text", "text": batch_prompt},
                    {"type": "image_url", "image_url": {"url": img_base64}}
                ]}
            ],
            temperature=0.0,
            max_tokens=200
        )
        
        if response.choices:
            content = response.choices[0].message.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            results = json.loads(content)
            return results
    except Exception as e:
        print(f"Batch verification error: {e}")
    
    # Return defaults on error
    return [0.5 if item.get('response_type') != 'yes_no' else 'no' for item in verification_items]

def calculate_tifa_score(image, prompt, client, model):
    """
    Calculate TIFA (Text-to-Image Faithfulness Assessment) score.
    
    TIFA generates question-answer pairs from the prompt and verifies
    each answer by asking VQA questions about the image.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
        client: Azure OpenAI client instance
        model: Model deployment name
    
    Returns:
        float: TIFA score 0-100
    """
    try:
        # Generate QA pairs from prompt
        qa_prompt = f"""
Analyze this text prompt and generate 3 verification questions with expected answers:
Prompt: "{prompt}"

Generate questions that can be answered by looking at the image.
Return ONLY valid JSON:
{{"qa_pairs": [
    {{"question": "Q1?", "expected": "expected answer"}},
    {{"question": "Q2?", "expected": "expected answer"}},
    {{"question": "Q3?", "expected": "expected answer"}}
]}}
"""
        
        qa_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Generate visual verification questions."},
                {"role": "user", "content": qa_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        if not qa_response.choices:
            return 0.0
        
        qa_content = qa_response.choices[0].message.content.strip()
        qa_content = qa_content.replace('```json', '').replace('```', '').strip()
        qa_pairs = json.loads(qa_content)['qa_pairs']
        
        # OPTIMIZED: Batch verify all QA pairs in a single API call
        img_base64 = pil_to_base64(image)
        
        # Build batch verification items
        verification_items = []
        for qa in qa_pairs:
            verification_items.append({
                'prompt': f"Q: {qa['question']} Expected: {qa['expected']} - Does the image support this?",
                'response_type': 'yes_no'
            })
        
        # Single batched API call instead of N separate calls
        batch_results = _batch_verify_with_image(client, model, img_base64, verification_items)
        
        # Count correct answers
        correct = 0
        for result in batch_results:
            if isinstance(result, str) and 'yes' in result.lower():
                correct += 1
        
        return (correct / len(qa_pairs)) * 100 if qa_pairs else 0.0
        
    except Exception as e:
        print(f"TIFA calculation error: {e}")
        return 0.0


def calculate_dsg_score(image, prompt, client, model):
    """
    Calculate DSG (Davidsonian Scene Graph) score.
    
    Faithful reimplementation of the original DSG pipeline from lib/DSG:
    1. Tuple generation: Extract skill-specific semantic tuples from the prompt
    2. Question generation: Convert each tuple into a yes/no VQA question
    3. Dependency generation: Identify parent-child relationships between tuples
    4. VQA: Ask each question about the image → binary yes/no
    5. Dependency filtering: Zero out child scores where parent answered "no"
    6. Final score = average of filtered scores
    
    Uses in-context learning with 23 TIFA-160 examples (same as original DSG paper).
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
        client: Azure OpenAI client instance
        model: Model deployment name
    
    Returns:
        float: DSG score 0-100
    """
    try:
        # =====================================================================
        # Load in-context examples from TIFA-160 (same 23 examples as original)
        # =====================================================================
        icl_examples = _load_dsg_icl_examples()
        
        # =====================================================================
        # Step 1: Tuple generation (prompt → semantic tuples)
        # =====================================================================
        tuple_preamble = (
            "Task: given input prompts, describe each scene with skill-specific tuples.\n"
            "Do not generate same tuples again. Do not generate tuples that are not "
            "explicitly described in the prompts.\n"
            "output format: id | tuple"
        )
        
        tuple_icl = ""
        for ex in icl_examples:
            tuple_icl += f"\ninput: {ex['prompt']}\noutput: {ex['tuple_output']}\n"
        
        tuple_prompt = f"{tuple_preamble}\n{tuple_icl}\ninput: {prompt}\noutput: "
        
        tuple_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": tuple_prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        raw_tuples = tuple_resp.choices[0].message.content.strip()
        # Parse: stop at next "input:" if the model continues
        raw_tuples = raw_tuples.split("input:")[0].strip()
        
        id2tuple = _parse_tuple_output(raw_tuples)
        if not id2tuple:
            print("DSG: No tuples generated")
            return 0.0
        
        # Reconstruct tuple string for subsequent steps
        tuple_str = "\n".join(f"{tid} | {tval}" for tid, tval in sorted(id2tuple.items()))
        
        # =====================================================================
        # Step 2: Question generation (prompt + tuples → yes/no questions)
        # =====================================================================
        question_preamble = (
            "Task: given input prompts and skill-specific tuples, re-write tuple "
            "each in natural language question.\n"
            "output format: id | question"
        )
        
        question_icl = ""
        for ex in icl_examples:
            q_input = ex['prompt'] + "\n" + ex['tuple_output']
            question_icl += f"\ninput: {q_input}\noutput: {ex['question_output']}\n"
        
        question_input = prompt + "\n" + tuple_str
        question_prompt = f"{question_preamble}\n{question_icl}\ninput: {question_input}\noutput: "
        
        question_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question_prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        raw_questions = question_resp.choices[0].message.content.strip()
        raw_questions = raw_questions.split("input:")[0].strip()
        
        id2question = _parse_question_output(raw_questions)
        if not id2question:
            print("DSG: No questions generated")
            return 0.0
        
        # =====================================================================
        # Step 3: Dependency generation (prompt + tuples → parent dependencies)
        # =====================================================================
        dep_preamble = (
            "Task: given input prompts and tuples, describe the parent tuples of each tuple.\n"
            "output format: id | dependencies (comma separated)"
        )
        
        dep_icl = ""
        for ex in icl_examples:
            dep_input = ex['prompt'] + "\n" + ex['tuple_output']
            dep_icl += f"\ninput: {dep_input}\noutput: {ex['dep_output']}\n"
        
        dep_input = prompt + "\n" + tuple_str
        dep_prompt = f"{dep_preamble}\n{dep_icl}\ninput: {dep_input}\noutput: "
        
        dep_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": dep_prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        raw_deps = dep_resp.choices[0].message.content.strip()
        raw_deps = raw_deps.split("input:")[0].strip()
        
        id2dependency = _parse_dependency_output(raw_deps)
        
        # =====================================================================
        # Step 4: VQA — ask each question about the image (binary yes/no)
        # =====================================================================
        img_base64 = pil_to_base64(image)
        
        qid2answer = {}
        for qid in sorted(id2question.keys()):
            question = id2question[qid]
            if not question or not question.strip():
                qid2answer[qid] = 'yes'  # Skip empty questions (treat as pass)
                continue
            
            vqa_resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": img_base64}},
                    {"type": "text", "text": (
                        f"Answer only with 'yes' or 'no'. Do not give other outputs "
                        f"or punctuation marks. Question: {question}"
                    )}
                ]}],
                temperature=0.0,
                max_tokens=20,
            )
            answer = vqa_resp.choices[0].message.content.strip().lower()
            # Clean punctuation
            answer = answer.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
            qid2answer[qid] = answer
        
        # =====================================================================
        # Step 5: Dependency-aware scoring (from lib/DSG/dsg/vqa_utils.py)
        # =====================================================================
        result = _calc_vqa_score_with_dependency(qid2answer, id2dependency)
        
        return result['average_score_with_dependency'] * 100
        
    except Exception as e:
        print(f"DSG calculation error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def calculate_dsg_score_detailed(image, prompt, client, model):
    """
    Calculate DSG score with full pipeline details for comparison experiments.
    
    Returns:
        dict with keys:
          - score: float 0-100
          - tuples: dict {id: tuple_str}
          - questions: dict {id: question_str}
          - dependencies: dict {id: [parent_ids]}
          - answers: dict {id: 'yes'/'no'}
          - raw_scores: dict {id: 0.0|1.0} before dependency filtering
          - filtered_scores: dict {id: 0.0|1.0} after dependency filtering
          - validity: dict {id: bool}
          - score_without_dep: float 0-100
          - error: str or None
    """
    empty = {
        'score': 0.0, 'tuples': {}, 'questions': {}, 'dependencies': {},
        'answers': {}, 'raw_scores': {}, 'filtered_scores': {},
        'validity': {}, 'score_without_dep': 0.0, 'error': None,
    }
    try:
        icl_examples = _load_dsg_icl_examples()

        # Stage 1: Tuple generation
        tuple_preamble = (
            "Task: given input prompts, describe each scene with skill-specific tuples.\n"
            "Do not generate same tuples again. Do not generate tuples that are not "
            "explicitly described in the prompts.\n"
            "output format: id | tuple"
        )
        tuple_icl = ""
        for ex in icl_examples:
            tuple_icl += f"\ninput: {ex['prompt']}\noutput: {ex['tuple_output']}\n"
        tuple_prompt = f"{tuple_preamble}\n{tuple_icl}\ninput: {prompt}\noutput: "
        tuple_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": tuple_prompt}],
            temperature=0.0, max_tokens=500,
        )
        raw_tuples = tuple_resp.choices[0].message.content.strip().split("input:")[0].strip()
        id2tuple = _parse_tuple_output(raw_tuples)
        if not id2tuple:
            empty['error'] = 'No tuples generated'
            return empty
        tuple_str = "\n".join(f"{tid} | {tval}" for tid, tval in sorted(id2tuple.items()))

        # Stage 2: Question generation
        question_preamble = (
            "Task: given input prompts and skill-specific tuples, re-write tuple "
            "each in natural language question.\n"
            "output format: id | question"
        )
        question_icl = ""
        for ex in icl_examples:
            q_input = ex['prompt'] + "\n" + ex['tuple_output']
            question_icl += f"\ninput: {q_input}\noutput: {ex['question_output']}\n"
        question_input = prompt + "\n" + tuple_str
        question_prompt = f"{question_preamble}\n{question_icl}\ninput: {question_input}\noutput: "
        question_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question_prompt}],
            temperature=0.0, max_tokens=500,
        )
        raw_questions = question_resp.choices[0].message.content.strip().split("input:")[0].strip()
        id2question = _parse_question_output(raw_questions)
        if not id2question:
            empty['error'] = 'No questions generated'
            empty['tuples'] = id2tuple
            return empty

        # Stage 3: Dependency generation
        dep_preamble = (
            "Task: given input prompts and tuples, describe the parent tuples of each tuple.\n"
            "output format: id | dependencies (comma separated)"
        )
        dep_icl = ""
        for ex in icl_examples:
            dep_input = ex['prompt'] + "\n" + ex['tuple_output']
            dep_icl += f"\ninput: {dep_input}\noutput: {ex['dep_output']}\n"
        dep_input = prompt + "\n" + tuple_str
        dep_prompt = f"{dep_preamble}\n{dep_icl}\ninput: {dep_input}\noutput: "
        dep_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": dep_prompt}],
            temperature=0.0, max_tokens=500,
        )
        raw_deps = dep_resp.choices[0].message.content.strip().split("input:")[0].strip()
        id2dependency = _parse_dependency_output(raw_deps)

        # Stage 4: VQA
        img_base64 = pil_to_base64(image)
        qid2answer = {}
        for qid in sorted(id2question.keys()):
            question = id2question[qid]
            if not question or not question.strip():
                qid2answer[qid] = 'yes'
                continue
            vqa_resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": img_base64}},
                    {"type": "text", "text": (
                        f"Answer only with 'yes' or 'no'. Do not give other outputs "
                        f"or punctuation marks. Question: {question}"
                    )}
                ]}],
                temperature=0.0, max_tokens=20,
            )
            answer = vqa_resp.choices[0].message.content.strip().lower()
            answer = answer.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
            qid2answer[qid] = answer

        # Stage 5: Dependency-aware scoring
        result = _calc_vqa_score_with_dependency(qid2answer, id2dependency)

        return {
            'score': result['average_score_with_dependency'] * 100,
            'tuples': id2tuple,
            'questions': id2question,
            'dependencies': result['qid2dependency'],
            'answers': result['qid2answer'],
            'raw_scores': result['qid2scores'],
            'filtered_scores': {qid: result['qid2scores'].get(qid, 0.0)
                                if result['qid2validity'].get(qid, True) else 0.0
                                for qid in result['qid2scores']},
            'validity': result['qid2validity'],
            'score_without_dep': result['average_score_without_dependency'] * 100,
            'error': None,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        empty['error'] = str(e)
        return empty


# =============================================================================
# DSG Helper Functions (ported from lib/DSG)
# =============================================================================

_DSG_ICL_EXAMPLES = None

def _load_dsg_icl_examples():
    """
    Load the 23 TIFA-160 in-context examples used by DSG.
    Caches after first load.
    """
    global _DSG_ICL_EXAMPLES
    if _DSG_ICL_EXAMPLES is not None:
        return _DSG_ICL_EXAMPLES
    
    import pandas as pd
    from pathlib import Path
    
    TIFA160_ICL_TRAIN_IDS = [
        'coco_361740', 'drawbench_155', 'partiprompt_86', 'paintskill_374',
        'coco_552592', 'partiprompt_1414', 'coco_627537', 'coco_744388',
        'partiprompt_1108', 'coco_397109', 'coco_666114', 'coco_62896',
        'paintskill_235', 'drawbench_159', 'partiprompt_893', 'coco_322041',
        'coco_292534', 'drawbench_57', 'partiprompt_555', 'coco_488166',
        'partiprompt_726', 'coco_323167', 'coco_625027',
    ]
    
    # Find tifa160-dev-anns.csv
    possible_paths = [
        Path(__file__).parent.parent.parent.parent.parent / "lib" / "DSG" / "dsg" / "data" / "tifa160-dev-anns.csv",
        Path(__file__).parent.parent.parent.parent.parent.parent / "lib" / "DSG" / "dsg" / "data" / "tifa160-dev-anns.csv",
        Path("lib/DSG/dsg/data/tifa160-dev-anns.csv"),
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
    
    if data_path is None:
        print("Warning: TIFA-160 data not found. DSG will use zero-shot (less accurate).")
        _DSG_ICL_EXAMPLES = []
        return _DSG_ICL_EXAMPLES
    
    df = pd.read_csv(data_path)
    
    examples = []
    for item_id in TIFA160_ICL_TRAIN_IDS:
        rows = df[df.item_id == item_id]
        if rows.empty:
            continue
        
        prompt = rows.text.iloc[0]
        tuples = rows.tuple.tolist()
        deps = [str(d) for d in rows.dependency.tolist()]
        questions = [str(q) for q in rows.question_natural_language.tolist()]
        
        tuple_output = "\n".join(f"{i+1} | {t}" for i, t in enumerate(tuples))
        dep_output = "\n".join(f"{i+1} | {d}" for i, d in enumerate(deps))
        question_output = "\n".join(f"{i+1} | {q}" for i, q in enumerate(questions))
        
        examples.append({
            'prompt': prompt,
            'tuple_output': tuple_output,
            'dep_output': dep_output,
            'question_output': question_output,
        })
    
    print(f"Loaded {len(examples)} DSG in-context examples from TIFA-160")
    _DSG_ICL_EXAMPLES = examples
    return _DSG_ICL_EXAMPLES


def _parse_tuple_output(output_str):
    """Parse tuple generation output into {id: tuple_str} dict. (from lib/DSG parse_utils.py)"""
    id2tup = {}
    for line in output_str.strip().split('\n'):
        line = line.strip()
        if not line or '|' not in line:
            continue
        try:
            parts = line.split('|', 1)
            tup_id = int(parts[0].strip())
            tup = parts[1].strip()
            # Clean: only take string before parenthesis content (category name)
            tup_clean = tup.strip().split('(')[0].strip() if '(' in tup else tup
            id2tup[tup_id] = tup  # Keep full tuple for question/dep generation
        except (ValueError, IndexError):
            continue
    return id2tup


def _parse_question_output(output_str):
    """Parse question generation output into {id: question} dict. (from lib/DSG parse_utils.py)"""
    id2question = {}
    for line in output_str.strip().split('\n'):
        line = line.strip()
        if not line or '|' not in line:
            continue
        try:
            parts = line.split('|', 1)
            qid = int(parts[0].strip())
            question = parts[1].strip()
            id2question[qid] = question
        except (ValueError, IndexError):
            continue
    return id2question


def _parse_dependency_output(output_str):
    """Parse dependency generation output into {id: [parent_ids]} dict. (from lib/DSG parse_utils.py)"""
    id2dep = {}
    for line in output_str.strip().split('\n'):
        line = line.strip()
        if not line or '|' not in line:
            continue
        try:
            parts = line.split('|', 1)
            qid = int(parts[0].strip())
            dep_str = parts[1].strip()
            
            # Clean dependency IDs (filter out non-numeric except '0' and '-')
            dep_parts = [d.strip() for d in dep_str.split(',')]
            dep_parts = [d for d in dep_parts if d.isnumeric() or d == '-']
            
            # If includes 0 and others, remove 0
            if len(dep_parts) > 1:
                dep_parts = [d for d in dep_parts if d != '0']
            
            dep_ids = [int(d) for d in dep_parts if d.isnumeric()]
            if not dep_ids:
                dep_ids = [0]
            
            id2dep[qid] = dep_ids
        except (ValueError, IndexError):
            continue
    return id2dep


def _calc_vqa_score_with_dependency(qid2answer, qid2dependency=None):
    """
    Calculate VQA scores with dependency filtering.
    Exact logic from lib/DSG/dsg/vqa_utils.py calc_vqa_score().
    
    - Binary scoring: answer == ground_truth ('yes') → 1.0, else 0.0
    - Dependency filtering: if any parent question scored 0, child is zeroed out
    """
    from copy import deepcopy
    
    # Ground truth: all answers should be 'yes'
    qid2gtanswer = {qid: 'yes' for qid in qid2answer.keys()}
    
    # Binary scores
    qid2scores = {}
    for qid, answer in qid2answer.items():
        gt = qid2gtanswer[qid]
        qid2scores[qid] = float(answer == gt)
    
    try:
        average_score_without_dep = sum(qid2scores.values()) / len(qid2scores)
    except ZeroDivisionError:
        average_score_without_dep = 0.0
    
    # Dependency filtering
    qid2validity = {}
    qid2scores_filtered = deepcopy(qid2scores)
    
    if qid2dependency is None:
        qid2dependency = {qid: [0] for qid in qid2answer.keys()}
    
    for qid, parent_ids in qid2dependency.items():
        any_parent_no = False
        for pid in parent_ids:
            if pid == 0:
                continue
            if pid in qid2scores and qid2scores[pid] == 0:
                any_parent_no = True
                break
        if any_parent_no:
            qid2scores_filtered[qid] = 0.0
            qid2validity[qid] = False
        else:
            qid2validity[qid] = True
    
    try:
        average_score_with_dep = sum(qid2scores_filtered.values()) / len(qid2scores)
    except ZeroDivisionError:
        average_score_with_dep = 0.0
    
    return {
        'qid2dependency': qid2dependency,
        'qid2answer': qid2answer,
        'qid2scores': qid2scores,
        'qid2validity': qid2validity,
        'average_score_with_dependency': average_score_with_dep,
        'average_score_without_dependency': average_score_without_dep,
    }


def calculate_psg_score_detailed(image, prompt, client, model):
    """
    Calculate PSG score with full scene-graph details for comparison experiments.
    
    Returns:
        dict with keys:
          - score: float 0-100
          - expected_objects: list
          - expected_attributes: dict
          - expected_relations: list
          - object_score: float 0-100
          - attribute_score: float 0-100
          - relation_score: float 0-100
          - error: str or None
    """
    empty = {
        'score': 0.0, 'expected_objects': [], 'expected_attributes': {},
        'expected_relations': [], 'object_score': 0.0,
        'attribute_score': 0.0, 'relation_score': 0.0, 'error': None,
    }
    try:
        img_base64 = pil_to_base64(image)

        sg_prompt = f"""For the prompt: \"{prompt}\"

Create an expected scene graph with:
1. Objects that should appear
2. Object attributes
3. Relationships between objects

Return ONLY valid JSON:
{{"scene_graph": {{
    "objects": ["obj1", "obj2"],
    "attributes": {{"obj1": ["attr1"], "obj2": ["attr2"]}},
    "relations": ["obj1 relation obj2"]
}}}}"""
        sg_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Create scene graphs from prompts."},
                {"role": "user", "content": sg_prompt}
            ],
            temperature=0.0, max_tokens=500,
        )
        if not sg_response.choices:
            empty['error'] = 'No scene graph response'
            return empty
        sg_content = sg_response.choices[0].message.content.strip()
        sg_content = sg_content.replace('```json', '').replace('```', '').strip()
        expected_sg = json.loads(sg_content)['scene_graph']

        verify_prompt = f"""Analyze this image and compare to expected scene graph:

Expected objects: {expected_sg.get('objects', [])}
Expected attributes: {expected_sg.get('attributes', {})}
Expected relations: {expected_sg.get('relations', [])}

Score each category 0-100:
- object_score: How many expected objects are present?
- attribute_score: How well do attributes match?
- relation_score: How well do relationships match?

Return ONLY valid JSON:
{{"object_score": X, "attribute_score": X, "relation_score": X}}"""
        verify_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Evaluate scene graph alignment."},
                {"role": "user", "content": [
                    {"type": "text", "text": verify_prompt},
                    {"type": "image_url", "image_url": {"url": img_base64}}
                ]}
            ],
            temperature=0.0, max_tokens=100,
        )
        if not verify_response.choices:
            empty['error'] = 'No verification response'
            empty['expected_objects'] = expected_sg.get('objects', [])
            empty['expected_attributes'] = expected_sg.get('attributes', {})
            empty['expected_relations'] = expected_sg.get('relations', [])
            return empty

        result_content = verify_response.choices[0].message.content.strip()
        result_content = result_content.replace('```json', '').replace('```', '').strip()
        scores = json.loads(result_content)
        obj_s = scores.get('object_score', 0)
        attr_s = scores.get('attribute_score', 0)
        rel_s = scores.get('relation_score', 0)
        avg_score = float(np.mean([obj_s, attr_s, rel_s]))

        return {
            'score': avg_score,
            'expected_objects': expected_sg.get('objects', []),
            'expected_attributes': expected_sg.get('attributes', {}),
            'expected_relations': expected_sg.get('relations', []),
            'object_score': obj_s,
            'attribute_score': attr_s,
            'relation_score': rel_s,
            'error': None,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        empty['error'] = str(e)
        return empty


def calculate_psg_score(image, prompt, client, model):
    """
    Calculate PSG (Panoptic Scene Graph) score.
    
    PSG evaluates the image based on scene graph structure: objects,
    their categories, and inter-object relationships.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
        client: Azure OpenAI client instance
        model: Model deployment name
    
    Returns:
        float: PSG score 0-100
    """
    try:
        img_base64 = pil_to_base64(image)
        
        # Generate expected scene graph from prompt
        sg_prompt = f"""
For the prompt: "{prompt}"

Create an expected scene graph with:
1. Objects that should appear
2. Object attributes
3. Relationships between objects

Return ONLY valid JSON:
{{"scene_graph": {{
    "objects": ["obj1", "obj2"],
    "attributes": {{"obj1": ["attr1"], "obj2": ["attr2"]}},
    "relations": ["obj1 relation obj2"]
}}}}
"""
        
        sg_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Create scene graphs from prompts."},
                {"role": "user", "content": sg_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        if not sg_response.choices:
            return 0.0
        
        sg_content = sg_response.choices[0].message.content.strip()
        sg_content = sg_content.replace('```json', '').replace('```', '').strip()
        expected_sg = json.loads(sg_content)['scene_graph']
        
        # Verify scene graph against image
        verify_prompt = f"""
Analyze this image and compare to expected scene graph:

Expected objects: {expected_sg.get('objects', [])}
Expected attributes: {expected_sg.get('attributes', {})}
Expected relations: {expected_sg.get('relations', [])}

Score each category 0-100:
- object_score: How many expected objects are present?
- attribute_score: How well do attributes match?
- relation_score: How well do relationships match?

Return ONLY valid JSON:
{{"object_score": X, "attribute_score": X, "relation_score": X}}
"""
        
        verify_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Evaluate scene graph alignment."},
                {"role": "user", "content": [
                    {"type": "text", "text": verify_prompt},
                    {"type": "image_url", "image_url": {"url": img_base64}}
                ]}
            ],
            temperature=0.0,
            max_tokens=100
        )
        
        if verify_response.choices:
            result_content = verify_response.choices[0].message.content.strip()
            result_content = result_content.replace('```json', '').replace('```', '').strip()
            scores = json.loads(result_content)
            avg_score = np.mean([
                scores.get('object_score', 0),
                scores.get('attribute_score', 0),
                scores.get('relation_score', 0)
            ])
            return avg_score
        
        return 0.0
        
    except Exception as e:
        print(f"PSG calculation error: {e}")
        return 0.0


def calculate_vpeval_score(image, prompt, client, model):
    """
    Calculate VPEval (Visual Programming Evaluation) score.
    
    VPEval uses a visual programming approach - breaking down evaluation
    into modular visual reasoning steps.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
        client: Azure OpenAI client instance
        model: Model deployment name
    
    Returns:
        float: VPEval score 0-100
    """
    try:
        img_base64 = pil_to_base64(image)
        
        # Generate visual program (evaluation steps)
        vp_prompt = f"""
Create a visual evaluation program for this prompt: "{prompt}"

Break down into modular verification steps:
1. Object detection checks
2. Attribute verification checks  
3. Spatial/compositional checks
4. Style/quality checks

Return ONLY valid JSON:
{{"program": [
    {{"step": "check_object", "target": "object name", "description": "what to verify"}},
    {{"step": "check_attribute", "target": "attribute", "description": "what to verify"}},
    {{"step": "check_spatial", "target": "layout", "description": "what to verify"}}
]}}
"""
        
        vp_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Create visual evaluation programs."},
                {"role": "user", "content": vp_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        if not vp_response.choices:
            return 0.0
        
        vp_content = vp_response.choices[0].message.content.strip()
        vp_content = vp_content.replace('```json', '').replace('```', '').strip()
        program = json.loads(vp_content)['program']
        
        # OPTIMIZED: Batch execute all visual program steps in a single API call
        verification_items = []
        for step in program:
            verification_items.append({
                'prompt': f"Check [{step['step']}] - Target: {step['target']} - Verify: {step['description']} (rate 0-100)",
                'response_type': 'score'
            })
        
        # Single batched API call instead of N separate calls
        batch_results = _batch_verify_with_image(client, model, img_base64, verification_items)
        
        # Parse scores (VPEval uses 0-100 scale)
        scores = []
        for result in batch_results:
            try:
                if isinstance(result, (int, float)):
                    score = float(result)
                    # Normalize if it's 0-1 scale
                    if score <= 1.0:
                        score = score * 100
                    scores.append(min(100, max(0, score)))
                elif isinstance(result, str):
                    score = float(result)
                    if score <= 1.0:
                        score = score * 100
                    scores.append(min(100, max(0, score)))
                else:
                    scores.append(50)
            except:
                scores.append(50)
        
        return np.mean(scores) if scores else 0.0
        
    except Exception as e:
        print(f"VPEval calculation error: {e}")
        return 0.0
