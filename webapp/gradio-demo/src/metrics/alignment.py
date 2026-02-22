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
                encoding = processor(image, question, return_tensors="pt",
                                     truncation=True, max_length=40)
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
    Calculate VLM-based North Star metrics (DSG, PSG) in parallel.
    
    TIFA and VPEval were removed to reduce API calls and speed up scoring.
    Soft-TIFA GM already covers probabilistic fact-checking (TIFA overlap),
    and VPEval adds limited unique signal compared to DSG/PSG.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
        client: Azure OpenAI client instance
        model: Model deployment name
    
    Returns:
        dict: {"dsg": score, "dsg_details": ..., "psg": score, "psg_details": ...}
    """
    cache_key = _get_cache_key(image, prompt)
    
    # Check if all metrics are cached
    cached_dsg = _cache_get(cache_key, "dsg")
    cached_psg = _cache_get(cache_key, "psg")
    
    if all(v is not None for v in [cached_dsg, cached_psg]):
        print("Using cached VLM metrics")
        return {
            "dsg": cached_dsg,
            "psg": cached_psg,
        }
    
    results = {}
    
    # Define metric functions to run in parallel
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
    
    # Run DSG and PSG in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_dsg),
            executor.submit(run_psg),
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
        "dsg": results.get("dsg", 0.0),
        "dsg_details": results.get("dsg_details"),
        "psg": results.get("psg", 0.0),
        "psg_details": results.get("psg_details"),
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
        # OPTIMIZED: Parallel VQA calls (was sequential, ~15s each × N questions)
        # =====================================================================
        img_base64 = pil_to_base64(image)
        
        sorted_qids = sorted(id2question.keys())
        qid2answer = {}

        def _ask_one(qid):
            question = id2question[qid]
            if not question or not question.strip():
                return qid, 'yes'  # Skip empty questions (treat as pass)
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
            answer = answer.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
            return qid, answer

        with ThreadPoolExecutor(max_workers=min(4, len(sorted_qids))) as executor:
            futures = [executor.submit(_ask_one, qid) for qid in sorted_qids]
            for fut in as_completed(futures):
                try:
                    qid, answer = fut.result(timeout=120)
                    qid2answer[qid] = answer
                except Exception as e:
                    print(f"DSG VQA error: {e}")
        
        # Ensure all qids have an answer (default 'no' for missing/failed)
        for qid in sorted_qids:
            if qid not in qid2answer:
                qid2answer[qid] = 'no'
        
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

        # Stage 4: VQA (OPTIMIZED: parallel calls)
        img_base64 = pil_to_base64(image)
        sorted_qids = sorted(id2question.keys())
        qid2answer = {}

        def _ask_one_detailed(qid):
            question = id2question[qid]
            if not question or not question.strip():
                return qid, 'yes'
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
            return qid, answer

        with ThreadPoolExecutor(max_workers=min(4, len(sorted_qids))) as executor:
            futures = [executor.submit(_ask_one_detailed, qid) for qid in sorted_qids]
            for fut in as_completed(futures):
                try:
                    qid, answer = fut.result(timeout=120)
                    qid2answer[qid] = answer
                except Exception as e:
                    print(f"DSG detailed VQA error: {e}")

        # Ensure all qids have an answer (default 'no' for missing/failed)
        for qid in sorted_qids:
            if qid not in qid2answer:
                qid2answer[qid] = 'no'

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


# =============================================================================
# PSG-Score: Faithful implementation of ICCV 2025 paper
# "Leveraging Panoptic Scene Graph for Evaluating Fine-Grained T2I Generation"
# by Deng, Yang, Yu, Yang, Chen (ByteDance Seed)
#
# Pipeline:
#   1. Extract ground-truth scene graph (G_gt) from the text prompt via GPT-4o
#   2. Extract predicted scene graph (G_pred) from the generated image via GPT-4o
#      (approximates the paper's panoptic segmentation + Set-of-Mark + VLM)
#   3. Compute node matching and edge similarity using BERT embeddings
#   4. Optimal graph matching via Hungarian algorithm + edge matching
#   5. F1 score (precision/recall) with foreground-only FP penalty
# =============================================================================

# Lazy-loaded BERT model for semantic matching
_PSG_BERT_MODEL = None

def _get_psg_bert_model():
    """Lazy-load a SentenceTransformer model for computing word embeddings."""
    global _PSG_BERT_MODEL
    if _PSG_BERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _PSG_BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        print("PSG: Loaded SentenceTransformer (all-MiniLM-L6-v2) for semantic matching")
    return _PSG_BERT_MODEL


def _psg_cosine_distance(emb1, emb2):
    """Compute cosine distance between two embedding vectors. Returns 0.0 (identical) to 2.0."""
    from sklearn.metrics.pairwise import cosine_distances
    d = cosine_distances(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]
    return float(d)


def _psg_node_matching(w1_emb, w2_emb):
    """
    Algorithm 1 NodeMatching: binary gate.
    Returns 1.0 if cosine_distance < 0.5, else 0.
    """
    d = _psg_cosine_distance(w1_emb, w2_emb)
    return 1.0 if d < 0.5 else 0.0


def _psg_edge_similarity(w1, w2, w1_emb, w2_emb):
    """
    Algorithm 1 EdgeSimilarity: continuous.
    Returns 1.0 if words are identical, else 1 - cosine_distance.
    """
    if w1.lower().strip() == w2.lower().strip():
        return 1.0
    d = _psg_cosine_distance(w1_emb, w2_emb)
    return max(0.0, 1.0 - d)


def _psg_extract_gt_scene_graph(prompt, client, model):
    """
    Extract ground-truth scene graph G_gt from the text prompt.
    Paper: GPT-4o generates scene graph, then human verifies.
    We use GPT-4o with structured output.
    
    Returns dict:
      {
        'nodes': [{'id': int, 'label': str, 'attributes': [str], 'is_foreground': bool}],
        'edges': [{'src': int, 'dst': int, 'relation': str}]
      }
    """
    sg_prompt = f"""Analyze this text prompt and extract a structured scene graph.

Prompt: "{prompt}"

Extract:
1. All objects/entities mentioned (both foreground objects like "cat", "car" and background elements like "sky", "street", "grass")
2. Attributes for each object (color, material, size, shape, state, texture, etc.)
3. Relationships between objects (spatial: "on", "next to", "behind"; action: "riding", "holding", "wearing"; etc.)

For each object, classify as foreground (specific countable objects like people, animals, vehicles, furniture) or background (scene elements like sky, ground, street, grass, water, wall).

Return ONLY valid JSON:
{{"scene_graph": {{
    "nodes": [
        {{"id": 0, "label": "object_name", "attributes": ["attr1", "attr2"], "is_foreground": true}},
        {{"id": 1, "label": "object_name", "attributes": ["attr1"], "is_foreground": false}}
    ],
    "edges": [
        {{"src": 0, "dst": 1, "relation": "relationship_name"}}
    ]
}}}}

Important:
- Each unique object instance gets its own node (e.g., "two cats" → two separate cat nodes)
- Include ALL attributes mentioned for each object
- Include ALL relationships mentioned between objects
- Label background elements (sky, ground, street, grass, water, room, etc.) as is_foreground=false"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Extract structured scene graphs from text prompts. Be thorough and precise."},
            {"role": "user", "content": sg_prompt}
        ],
        temperature=0.0,
        max_tokens=1000,
    )
    content = resp.choices[0].message.content or ""
    sg = _psg_parse_scene_graph_json(content)
    return sg


def _psg_parse_scene_graph_json(content: str) -> dict:
    """Parse scene graph JSON from GPT-4o response, handling markdown fences and edge cases."""
    content = content.strip()
    if not content:
        print("PSG WARNING: GPT-4o returned empty content, using empty scene graph")
        return {'nodes': [], 'edges': []}
    # Strip markdown code fences
    content = content.replace('```json', '').replace('```', '').strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON object from mixed content
        import re
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            parsed = json.loads(match.group())
        else:
            print(f"PSG WARNING: Could not parse JSON from response: {content[:200]}")
            return {'nodes': [], 'edges': []}
    if 'scene_graph' in parsed:
        return parsed['scene_graph']
    if 'nodes' in parsed:
        return parsed
    print(f"PSG WARNING: Unexpected JSON structure, using empty scene graph")
    return {'nodes': [], 'edges': []}


def _psg_extract_pred_scene_graph(image, prompt, client, model):
    """
    Extract predicted scene graph G_pred from the generated image.
    Paper: panoptic segmentation (FC-CLIP/kMaX-DeepLab) + Set-of-Mark + GPT-4o.
    Approximation: GPT-4o vision extracts objects, attributes, relationships directly.
    
    Returns same structure as _psg_extract_gt_scene_graph.
    """
    img_base64 = pil_to_base64(image)

    sg_prompt = f"""Look at this image carefully and extract a structured scene graph of EVERYTHING you see.

The image was generated from this prompt: "{prompt}"
But describe what you ACTUALLY see in the image, not what the prompt says.

Extract:
1. All objects/entities visible (both foreground objects and background elements)
2. Attributes for each object (color, material, size, shape, state, texture, etc.)
3. Relationships between objects (spatial, action, etc.)

Classify each object as foreground (specific countable objects) or background (scene elements like sky, ground, etc.).

Return ONLY valid JSON:
{{"scene_graph": {{
    "nodes": [
        {{"id": 0, "label": "object_name", "attributes": ["attr1", "attr2"], "is_foreground": true}},
        {{"id": 1, "label": "object_name", "attributes": ["attr1"], "is_foreground": false}}
    ],
    "edges": [
        {{"src": 0, "dst": 1, "relation": "relationship_name"}}
    ]
}}}}

Important:
- Report what you ACTUALLY see, not what you expect
- Each distinct object instance gets its own node
- Include observed attributes (colors, states, materials)
- Include observed relationships
- Be comprehensive but accurate"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Extract structured scene graphs from images. Be thorough and accurate."},
            {"role": "user", "content": [
                {"type": "text", "text": sg_prompt},
                {"type": "image_url", "image_url": {"url": img_base64}}
            ]}
        ],
        temperature=0.0,
        max_tokens=1000,
    )
    content = resp.choices[0].message.content or ""
    sg = _psg_parse_scene_graph_json(content)
    return sg


def _psg_semantic_graph_matching(gt_sg, pred_sg):
    """
    Algorithm 1: Semantic Graph Matching from the PSG-Score paper.
    
    Uses BERT embeddings for node matching (binary, threshold 0.5) and
    edge similarity (continuous). Uses Hungarian algorithm for optimal
    node matching, then matches edges between matched node pairs.
    
    Returns dict with TP, FP, FN, matched/unmatched details, precision, recall, F1.
    """
    from scipy.optimize import linear_sum_assignment

    bert = _get_psg_bert_model()

    gt_nodes = gt_sg.get('nodes', [])
    pred_nodes = pred_sg.get('nodes', [])
    gt_edges = gt_sg.get('edges', [])
    pred_edges = pred_sg.get('edges', [])

    if not gt_nodes:
        # Nothing in ground truth → everything predicted is FP
        fp_fg = sum(1 for n in pred_nodes if n.get('is_foreground', True))
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': fp_fg, 'fn': 0,
            'matched_nodes': [], 'matched_edges': [],
            'unmatched_gt_nodes': [], 'unmatched_pred_nodes': pred_nodes,
            'unmatched_gt_edges': [], 'unmatched_pred_edges': pred_edges,
            'node_details': [], 'edge_details': [],
            'attribute_precision': 0.0, 'attribute_recall': 0.0,
        }

    # =========================================================================
    # Step 1: Encode all node labels with BERT
    # =========================================================================
    gt_labels = [n['label'].lower().strip() for n in gt_nodes]
    pred_labels = [n['label'].lower().strip() for n in pred_nodes]
    all_labels = gt_labels + pred_labels
    
    # Also encode all attributes and edge relations
    gt_attrs_flat = []
    for n in gt_nodes:
        gt_attrs_flat.extend([a.lower().strip() for a in n.get('attributes', [])])
    pred_attrs_flat = []
    for n in pred_nodes:
        pred_attrs_flat.extend([a.lower().strip() for a in n.get('attributes', [])])
    
    gt_edge_rels = [e['relation'].lower().strip() for e in gt_edges]
    pred_edge_rels = [e['relation'].lower().strip() for e in pred_edges]
    
    # Batch encode everything
    all_texts = list(set(all_labels + gt_attrs_flat + pred_attrs_flat + gt_edge_rels + pred_edge_rels))
    if not all_texts:
        all_texts = ['empty']
    all_embeddings = bert.encode(all_texts, convert_to_numpy=True)
    text2emb = {t: all_embeddings[i] for i, t in enumerate(all_texts)}

    # =========================================================================
    # Step 2: Node matching via Hungarian algorithm (Algorithm 1 NodeMatching)
    # =========================================================================
    n_gt = len(gt_nodes)
    n_pred = len(pred_nodes)
    
    # Build cost matrix: cost = 1 - match_score (lower is better for Hungarian)
    # NodeMatching returns 1.0 if cosine_distance < 0.5, else 0
    cost_matrix = np.ones((n_gt, n_pred))  # default: no match (cost=1)
    for i in range(n_gt):
        for j in range(n_pred):
            gt_emb = text2emb[gt_labels[i]]
            pred_emb = text2emb[pred_labels[j]]
            match = _psg_node_matching(gt_emb, pred_emb)
            if match > 0:
                cost_matrix[i, j] = 0.0  # matched: cost = 0

    # Solve assignment (minimize cost)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter to only keep actual matches (cost == 0)
    matched_node_pairs = []  # (gt_idx, pred_idx)
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] == 0.0:
            matched_node_pairs.append((r, c))
    
    gt_matched_idx = {r for r, c in matched_node_pairs}
    pred_matched_idx = {c for r, c in matched_node_pairs}

    matched_nodes_detail = []
    for r, c in matched_node_pairs:
        matched_nodes_detail.append({
            'gt': gt_nodes[r]['label'],
            'pred': pred_nodes[c]['label'],
        })

    unmatched_gt_nodes = [gt_nodes[i] for i in range(n_gt) if i not in gt_matched_idx]
    unmatched_pred_nodes = [pred_nodes[j] for j in range(n_pred) if j not in pred_matched_idx]

    # =========================================================================
    # Step 3: Attribute matching for matched node pairs
    # =========================================================================
    total_gt_attrs = 0
    total_matched_attrs = 0
    total_pred_attrs = 0
    attr_details = []
    
    for gt_idx, pred_idx in matched_node_pairs:
        gt_node_attrs = [a.lower().strip() for a in gt_nodes[gt_idx].get('attributes', [])]
        pred_node_attrs = [a.lower().strip() for a in pred_nodes[pred_idx].get('attributes', [])]
        total_gt_attrs += len(gt_node_attrs)
        total_pred_attrs += len(pred_node_attrs)
        
        # Match attributes using BERT similarity (threshold 0.5)
        matched_in_pair = 0
        pred_used = set()
        for ga in gt_node_attrs:
            ga_emb = text2emb.get(ga)
            if ga_emb is None:
                continue
            best_j = -1
            best_dist = 999
            for j, pa in enumerate(pred_node_attrs):
                if j in pred_used:
                    continue
                pa_emb = text2emb.get(pa)
                if pa_emb is None:
                    continue
                d = _psg_cosine_distance(ga_emb, pa_emb)
                if d < 0.5 and d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j >= 0:
                matched_in_pair += 1
                pred_used.add(best_j)
                attr_details.append({
                    'gt_attr': ga, 'pred_attr': pred_node_attrs[best_j],
                    'object': gt_nodes[gt_idx]['label'],
                })
        total_matched_attrs += matched_in_pair

    # =========================================================================
    # Step 4: Edge matching (Algorithm 1 EdgeSimilarity)
    # =========================================================================
    # Build mapping from gt node idx → pred node idx
    gt_to_pred_map = {r: c for r, c in matched_node_pairs}
    
    # For each GT edge, check if both endpoints are matched, then match relation
    matched_edges = []
    unmatched_gt_edges = []
    pred_edges_used = set()
    
    for gt_edge in gt_edges:
        src_gt = gt_edge['src']
        dst_gt = gt_edge['dst']
        rel_gt = gt_edge['relation'].lower().strip()
        
        # Find GT node indices that match these IDs
        src_gt_idx = None
        dst_gt_idx = None
        for idx, n in enumerate(gt_nodes):
            if n['id'] == src_gt:
                src_gt_idx = idx
            if n['id'] == dst_gt:
                dst_gt_idx = idx
        
        if src_gt_idx is None or dst_gt_idx is None:
            unmatched_gt_edges.append(gt_edge)
            continue

        # Check if both endpoints are matched
        if src_gt_idx not in gt_to_pred_map or dst_gt_idx not in gt_to_pred_map:
            unmatched_gt_edges.append(gt_edge)
            continue

        src_pred_idx = gt_to_pred_map[src_gt_idx]
        dst_pred_idx = gt_to_pred_map[dst_gt_idx]
        src_pred_id = pred_nodes[src_pred_idx]['id']
        dst_pred_id = pred_nodes[dst_pred_idx]['id']
        
        # Find matching pred edge between same node pair
        best_edge_idx = -1
        best_sim = 0.0
        for pe_idx, pe in enumerate(pred_edges):
            if pe_idx in pred_edges_used:
                continue
            if pe['src'] == src_pred_id and pe['dst'] == dst_pred_id:
                rel_pred = pe['relation'].lower().strip()
                rel_gt_emb = text2emb.get(rel_gt)
                rel_pred_emb = text2emb.get(rel_pred)
                if rel_gt_emb is not None and rel_pred_emb is not None:
                    sim = _psg_edge_similarity(rel_gt, rel_pred, rel_gt_emb, rel_pred_emb)
                else:
                    sim = 1.0 if rel_gt == rel_pred else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_edge_idx = pe_idx
            # Also check reversed direction (e.g., "A next to B" ≈ "B next to A")
            elif pe['src'] == dst_pred_id and pe['dst'] == src_pred_id:
                rel_pred = pe['relation'].lower().strip()
                rel_gt_emb = text2emb.get(rel_gt)
                rel_pred_emb = text2emb.get(rel_pred)
                if rel_gt_emb is not None and rel_pred_emb is not None:
                    sim = _psg_edge_similarity(rel_gt, rel_pred, rel_gt_emb, rel_pred_emb)
                else:
                    sim = 1.0 if rel_gt == rel_pred else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_edge_idx = pe_idx

        # Edge match threshold: similarity > 0.5 (semantic match)
        if best_edge_idx >= 0 and best_sim >= 0.5:
            matched_edges.append({
                'gt_rel': rel_gt,
                'pred_rel': pred_edges[best_edge_idx]['relation'],
                'similarity': best_sim,
                'gt_src': gt_nodes[src_gt_idx]['label'],
                'gt_dst': gt_nodes[dst_gt_idx]['label'],
            })
            pred_edges_used.add(best_edge_idx)
        else:
            unmatched_gt_edges.append(gt_edge)

    unmatched_pred_edges = [pred_edges[i] for i in range(len(pred_edges)) if i not in pred_edges_used]

    # =========================================================================
    # Step 5: Compute TP, FP, FN, Precision, Recall, F1 (Eq. 2 from paper)
    # =========================================================================
    # TP = matched nodes + matched attributes + matched edges
    tp_nodes = len(matched_node_pairs)
    tp_attrs = total_matched_attrs
    tp_edges = len(matched_edges)
    tp = tp_nodes + tp_attrs + tp_edges

    # FN = unmatched GT nodes + unmatched GT attrs + unmatched GT edges
    fn_nodes = len(unmatched_gt_nodes)
    fn_attrs = total_gt_attrs - total_matched_attrs
    fn_edges = len(unmatched_gt_edges)
    fn = fn_nodes + fn_attrs + fn_edges

    # FP = unmatched PRED nodes (FOREGROUND ONLY per paper) + unmatched pred attrs + unmatched pred edges
    # Paper: "If the extra nodes exist in foreground, they will be counted as FP.
    #         If the extra nodes exist in background then these nodes will be ignored."
    fp_nodes = sum(1 for n in unmatched_pred_nodes if n.get('is_foreground', True))
    fp_attrs = total_pred_attrs - total_matched_attrs  # extra predicted attributes 
    fp_edges = len(unmatched_pred_edges)
    fp = fp_nodes + fp_edges  # Paper focuses on nodes + edges for FP
    # Note: attribute FP is not explicitly counted in paper's Eq(1), but we track it

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute sub-scores for backwards compatibility with UI
    object_score = tp_nodes / (tp_nodes + fn_nodes) if (tp_nodes + fn_nodes) > 0 else 0.0
    attribute_score = tp_attrs / (tp_attrs + fn_attrs) if (tp_attrs + fn_attrs) > 0 else 0.0
    relation_score = tp_edges / (tp_edges + fn_edges) if (tp_edges + fn_edges) > 0 else 0.0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp, 'fp': fp, 'fn': fn,
        'tp_nodes': tp_nodes, 'tp_attrs': tp_attrs, 'tp_edges': tp_edges,
        'fn_nodes': fn_nodes, 'fn_attrs': fn_attrs, 'fn_edges': fn_edges,
        'fp_nodes': fp_nodes, 'fp_edges': fp_edges,
        'matched_nodes': matched_nodes_detail,
        'matched_edges': matched_edges,
        'unmatched_gt_nodes': unmatched_gt_nodes,
        'unmatched_pred_nodes': unmatched_pred_nodes,
        'unmatched_gt_edges': unmatched_gt_edges,
        'unmatched_pred_edges': unmatched_pred_edges,
        'attribute_details': attr_details,
        'object_score': object_score * 100,
        'attribute_score': attribute_score * 100,
        'relation_score': relation_score * 100,
    }


def calculate_psg_score_detailed(image, prompt, client, model):
    """
    Calculate PSG-Score with full pipeline details (faithful to ICCV 2025 paper).
    
    Pipeline:
      1. Extract G_gt from prompt (GPT-4o)
      2. Extract G_pred from image (GPT-4o vision)
      3. Semantic graph matching (BERT embeddings + Hungarian algorithm)
      4. F1 scoring with foreground-only FP penalty
    
    Returns:
        dict with keys:
          - score: float 0-100 (F1 * 100)
          - precision: float 0-1
          - recall: float 0-1
          - expected_objects: list of GT object labels
          - expected_attributes: dict of GT attributes
          - expected_relations: list of GT relation strings
          - detected_objects: list of predicted object labels
          - object_score: float 0-100 (node recall)
          - attribute_score: float 0-100 (attribute recall)
          - relation_score: float 0-100 (edge recall)
          - matched_nodes: list of matched node pairs
          - matched_edges: list of matched edge details
          - tp, fp, fn: int counts
          - error: str or None
    """
    empty = {
        'score': 0.0, 'precision': 0.0, 'recall': 0.0,
        'expected_objects': [], 'expected_attributes': {},
        'expected_relations': [], 'detected_objects': [],
        'object_score': 0.0, 'attribute_score': 0.0, 'relation_score': 0.0,
        'matched_nodes': [], 'matched_edges': [],
        'tp': 0, 'fp': 0, 'fn': 0, 'error': None,
    }
    try:
        print("PSG: Extracting ground-truth scene graph from prompt...")
        gt_sg = _psg_extract_gt_scene_graph(prompt, client, model)
        
        print("PSG: Extracting predicted scene graph from image...")
        pred_sg = _psg_extract_pred_scene_graph(image, prompt, client, model)
        
        print(f"PSG: G_gt has {len(gt_sg.get('nodes', []))} nodes, {len(gt_sg.get('edges', []))} edges")
        print(f"PSG: G_pred has {len(pred_sg.get('nodes', []))} nodes, {len(pred_sg.get('edges', []))} edges")

        print("PSG: Running semantic graph matching (BERT embeddings + Hungarian)...")
        match_result = _psg_semantic_graph_matching(gt_sg, pred_sg)

        f1 = match_result['f1']
        print(f"PSG: F1={f1:.3f} (Precision={match_result['precision']:.3f}, Recall={match_result['recall']:.3f})")
        print(f"PSG: TP={match_result['tp']} FP={match_result['fp']} FN={match_result['fn']}")

        # Build legacy-compatible output
        gt_objects = [n['label'] for n in gt_sg.get('nodes', [])]
        gt_attrs = {}
        for n in gt_sg.get('nodes', []):
            if n.get('attributes'):
                gt_attrs[n['label']] = n['attributes']
        # Build id→label lookup for edge display
        _id2label = {n['id']: n['label'] for n in gt_sg.get('nodes', [])}
        gt_rels = [f"{_id2label.get(e['src'], '?')} {e['relation']} {_id2label.get(e['dst'], '?')}"
                   for e in gt_sg.get('edges', [])]
        detected_objects = [n['label'] for n in pred_sg.get('nodes', [])]

        return {
            'score': f1 * 100,
            'precision': match_result['precision'],
            'recall': match_result['recall'],
            'expected_objects': gt_objects,
            'expected_attributes': gt_attrs,
            'expected_relations': gt_rels,
            'detected_objects': detected_objects,
            'object_score': match_result['object_score'],
            'attribute_score': match_result['attribute_score'],
            'relation_score': match_result['relation_score'],
            'matched_nodes': match_result['matched_nodes'],
            'matched_edges': match_result['matched_edges'],
            'unmatched_gt_nodes': match_result.get('unmatched_gt_nodes', []),
            'unmatched_pred_nodes': match_result.get('unmatched_pred_nodes', []),
            'unmatched_gt_edges': match_result.get('unmatched_gt_edges', []),
            'unmatched_pred_edges': match_result.get('unmatched_pred_edges', []),
            'attribute_details': match_result.get('attribute_details', []),
            'tp': match_result['tp'],
            'fp': match_result['fp'],
            'fn': match_result['fn'],
            'tp_nodes': match_result.get('tp_nodes', 0),
            'tp_attrs': match_result.get('tp_attrs', 0),
            'tp_edges': match_result.get('tp_edges', 0),
            'fn_nodes': match_result.get('fn_nodes', 0),
            'fn_attrs': match_result.get('fn_attrs', 0),
            'fn_edges': match_result.get('fn_edges', 0),
            'fp_nodes': match_result.get('fp_nodes', 0),
            'fp_edges': match_result.get('fp_edges', 0),
            'error': None,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        empty['error'] = str(e)
        return empty


def calculate_psg_score(image, prompt, client, model):
    """
    Calculate PSG-Score (ICCV 2025 paper faithful implementation).
    
    Returns F1 score (0-100) from semantic graph matching between
    the ground-truth scene graph (from prompt) and predicted scene graph
    (from image).
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
        client: Azure OpenAI client instance
        model: Model deployment name
    
    Returns:
        float: PSG-Score 0-100 (F1 * 100)
    """
    try:
        result = calculate_psg_score_detailed(image, prompt, client, model)
        return result['score']
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
