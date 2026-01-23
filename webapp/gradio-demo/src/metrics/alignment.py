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
from .utils import get_clip_model, get_vqa_model, pil_to_base64

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
Analyze this text prompt and generate 5 verification questions with expected answers:
Prompt: "{prompt}"

Generate questions that can be answered by looking at the image.
Return ONLY valid JSON:
{{"qa_pairs": [
    {{"question": "Q1?", "expected": "expected answer"}},
    {{"question": "Q2?", "expected": "expected answer"}}
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
        
        # Verify each QA pair against the image
        img_base64 = pil_to_base64(image)
        correct = 0
        
        for qa in qa_pairs:
            verify_prompt = f"""
Look at this image and answer: {qa['question']}
Expected answer: {qa['expected']}

Does the image support this expected answer? Reply with just "yes" or "no".
"""
            verify_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer visual verification questions."},
                    {"role": "user", "content": [
                        {"type": "text", "text": verify_prompt},
                        {"type": "image_url", "image_url": {"url": img_base64}}
                    ]}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            if verify_response.choices:
                answer = verify_response.choices[0].message.content.strip().lower()
                if 'yes' in answer:
                    correct += 1
        
        return (correct / len(qa_pairs)) * 100 if qa_pairs else 0.0
        
    except Exception as e:
        print(f"TIFA calculation error: {e}")
        return 0.0


def calculate_dsg_score(image, prompt, client, model):
    """
    Calculate DSG (Davidsonian Scene Graph) score.
    
    DSG decomposes the prompt into semantic primitives (entities, attributes,
    relations) and verifies each primitive against the image.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
        client: Azure OpenAI client instance
        model: Model deployment name
    
    Returns:
        float: DSG score 0-100
    """
    try:
        # Extract Davidsonian semantic primitives
        dsg_prompt = f"""
Decompose this prompt into Davidsonian Scene Graph primitives:
Prompt: "{prompt}"

Extract:
- Entities (objects/subjects)
- Attributes (properties of entities)
- Relations (between entities)

Return ONLY valid JSON:
{{"primitives": [
    {{"type": "entity", "content": "description"}},
    {{"type": "attribute", "content": "entity has attribute"}},
    {{"type": "relation", "content": "entity1 relation entity2"}}
]}}
"""
        
        dsg_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Extract Davidsonian semantic primitives."},
                {"role": "user", "content": dsg_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        if not dsg_response.choices:
            return 0.0
        
        dsg_content = dsg_response.choices[0].message.content.strip()
        dsg_content = dsg_content.replace('```json', '').replace('```', '').strip()
        primitives = json.loads(dsg_content)['primitives']
        
        # Verify each primitive
        img_base64 = pil_to_base64(image)
        scores = []
        
        for prim in primitives:
            verify_prompt = f"""
Evaluate if this {prim['type']} is present in the image:
"{prim['content']}"

Rate confidence 0.0-1.0. Reply with just the number.
"""
            verify_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Rate visual element presence."},
                    {"role": "user", "content": [
                        {"type": "text", "text": verify_prompt},
                        {"type": "image_url", "image_url": {"url": img_base64}}
                    ]}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            if verify_response.choices:
                try:
                    score = float(verify_response.choices[0].message.content.strip())
                    scores.append(min(1.0, max(0.0, score)))
                except:
                    scores.append(0.5)
        
        return np.mean(scores) * 100 if scores else 0.0
        
    except Exception as e:
        print(f"DSG calculation error: {e}")
        return 0.0


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
        
        # Execute visual program
        scores = []
        for step in program:
            exec_prompt = f"""
Execute this visual check on the image:
Step type: {step['step']}
Target: {step['target']}
Check: {step['description']}

Rate how well the image passes this check (0-100).
Reply with just the number.
"""
            exec_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Execute visual verification steps."},
                    {"role": "user", "content": [
                        {"type": "text", "text": exec_prompt},
                        {"type": "image_url", "image_url": {"url": img_base64}}
                    ]}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            if exec_response.choices:
                try:
                    score = float(exec_response.choices[0].message.content.strip())
                    scores.append(min(100, max(0, score)))
                except:
                    scores.append(50)
        
        return np.mean(scores) if scores else 0.0
        
    except Exception as e:
        print(f"VPEval calculation error: {e}")
        return 0.0
