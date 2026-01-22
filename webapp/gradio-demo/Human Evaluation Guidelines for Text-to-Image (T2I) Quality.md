# Human Evaluation Guidelines for Text-to-Image (T2I) Quality

## Purpose
This document provides standardized guidance for human evaluators assessing the quality of images generated from text prompts.

The primary objective is to evaluate **prompt faithfulness and correctness**, not aesthetic appeal or personal preference. These guidelines are intentionally structured so they can be reused as instructions for automated or LLM-based evaluators.

---

## Evaluation Principle (Read First)
Evaluate **what is generated**, not what you think was intended.

Do **not**:
- Infer missing details
- Fill in intent beyond the prompt
- Judge based on personal taste or artistic preference

Do:
- Judge strictly against the text prompt
- Penalize missing, incorrect, or hallucinated elements

---

## Step 1: Prompt Decomposition
Before viewing the image, decompose the prompt into **atomic requirements**.

Identify:
- Objects (what must appear)
- Attributes (color, size, material, count)
- Actions (running, holding, open/closed)
- Relationships (left/right, on top of, behind)
- Style or modality (if explicitly specified)

### Example
**Prompt**:  
> A red ceramic mug on a wooden table with steam rising, photographed in natural morning light.

**Atomic Requirements**:
- A mug is present
- Mug is red
- Mug is ceramic
- Mug is on a wooden table
- Steam is visible
- Lighting appears natural and morning-like
- Photographic realism

---

## Step 2: Core Evaluation Dimensions

### 1. Instruction Faithfulness (Highest Priority)
Assess whether **all atomic requirements** are satisfied.

Check:
- Required objects are present
- Attributes match the prompt
- Relationships are correct
- Actions are accurately depicted

**Any missing or incorrect critical requirement constitutes a failure.**

---

### 2. Physical & Logical Plausibility
Unless the prompt explicitly allows fantasy or surrealism, verify:
- Correct anatomy and proportions
- Plausible physics (gravity, support, reflections)
- Logical object interactions

Evaluate consistency with the **prompt’s intended world**.

---

### 3. Visual Clarity & Recognizability
Determine whether:
- Key objects are clearly visible
- Important elements are not obscured
- The main subject can be easily identified

This dimension is about **interpretability**, not image quality or resolution.

---

### 4. Style & Modality Compliance
Only evaluate style if it is **explicitly stated** in the prompt.

Examples:
- Photorealistic
- Illustration
- Anime style
- Technical diagram

Do **not** penalize style if none is specified.

---

### 5. Absence of Hallucinated Content
Check for:
- Extra people, objects, or text not requested
- Watermarks, logos, or symbols not mentioned
- Additions that materially alter the scene meaning

Minor background elements are acceptable **only if they do not affect prompt satisfaction**.

---

## Step 3: Error Severity Guidelines

### Critical Errors (Fail)
- Missing required objects
- Incorrect object count
- Incorrect key attributes (color, species, identity)
- Incorrect relationships or actions

### Minor Errors (Partial Credit)
- Slight style mismatch
- Minor lighting inconsistencies
- Non-essential background inaccuracies

### Acceptable Variations
- Artistic interpretation where the prompt is underspecified
- Natural variation in pose, texture, or composition

---

## Step 4: Scoring Scheme

### Option A: Binary Scoring
- **Pass**: All critical requirements satisfied
- **Fail**: Any critical requirement violated

### Option B: Structured Scoring (Recommended)

| Dimension                    | Score |
|-----------------------------|-------|
| Instruction Faithfulness    | ⬜    |
| Physical Plausibility       | ⬜    |
| Visual Clarity              | ⬜    |
| Style Compliance            | ⬜    |
| No Hallucinated Content     | ⬜    |

Instruction Faithfulness should carry the **highest weight** in any aggregated score.

---

## Step 5: Bias & Consistency Rules
Evaluators must:
- Ignore personal aesthetic preferences
- Ignore model identity or reputation
- Ignore prompt difficulty
- Base decisions only on visible evidence

Two evaluators reviewing the same prompt and image should reach similar conclusions.

---

## Final Checklist for Evaluators
Before submitting your evaluation:
- Did I verify every atomic requirement?
- Did I penalize missing or incorrect elements?
- Did I avoid guessing intent?
- Did I ignore my personal preferences?
- Would another evaluator reasonably agree with my judgment?

---

## Notes for Large-Scale Evaluation (Optional)
- Use example-based calibration
- Periodically measure inter-rater agreement
- Include anchor examples for pass/fail thresholds
- Recalibrate evaluators regularly

---
