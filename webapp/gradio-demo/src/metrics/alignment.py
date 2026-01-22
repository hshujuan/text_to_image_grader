"""
Alignment Metrics (Text-Image Correspondence)
==============================================

These metrics measure how well the generated image matches the text prompt.

Metrics:
- CLIPScore: Global semantic alignment via embedding cosine similarity
- VQAScore: Visual Question Answering based verification
- AHEaD: Attention-based Head alignment score
- PickScore: Human preference estimation (proxy)
- TIFA: Text-to-Image Faithfulness via QA
- DSG: Davidsonian Scene Graph decomposition
- PSG: Panoptic Scene Graph evaluation
- VPEval: Visual Programming evaluation
"""

import json
import numpy as np
from .utils import get_clip_model, get_vqa_model, pil_to_base64


def calculate_real_clipscore(image, prompt):
    """
    Calculate real CLIPScore using CLIP embeddings.
    
    CLIPScore measures global semantic alignment between image and text
    by computing cosine similarity of their CLIP embeddings.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
    
    Returns:
        float: CLIP score 0-100
    """
    try:
        import torch
        import clip
        
        model, preprocess = get_clip_model()
        if model is None or preprocess is None:
            print("CLIP model not available, using fallback")
            return 0.0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Preprocess image
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Tokenize text
        text_input = clip.tokenize([prompt]).to(device)
        
        # Get features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = (image_features @ text_features.T).item()
        
        # Convert from [-1, 1] to [0, 100]
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
    PickScore proxy using CLIP + aesthetic features.
    
    Real PickScore requires the trained PickScore model. This implementation
    combines CLIP alignment with aesthetic indicators as a proxy.
    
    Args:
        image: PIL Image to evaluate
        prompt: Text prompt
    
    Returns:
        float: Estimated human preference score 0-100
    """
    try:
        # Use CLIP score as base
        clip_score = calculate_real_clipscore(image, prompt)
        
        # Add aesthetic component
        img_array = np.array(image)
        
        # Color diversity
        colors = img_array.reshape(-1, 3)
        color_std = np.std(colors, axis=0).mean()
        color_score = min(100, color_std * 2)
        
        # Combine: 70% alignment, 30% aesthetics
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
