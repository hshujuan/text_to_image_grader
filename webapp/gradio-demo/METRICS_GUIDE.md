# 📊 Metrics Guide: Understanding Your Image Quality Report

## Quick Overview

Your report contains **three types** of metrics, each using different evaluation methods:

| Icon | Type | Method | Examples |
|------|------|--------|----------|
| 🤖 | **Model** | Uses trained ML models | CLIP, ViLT, VQA |
| 📐 | **Code** | Uses mathematical algorithms | BRISQUE, NIQE |
| 🔍 | **VLM** | Uses GPT-4o vision analysis | Safety checks, qualitative assessment |

---

## 📈 Metrics Breakdown

### ⭐ NORTH STAR METRIC

#### **Soft-TIFA GM** (Geometric Mean)
- **What it measures**: Overall text-image alignment accuracy
- **How it works**: 
  1. GPT-4o extracts atomic facts from your prompt (e.g., "blue hair", "woman", "portrait")
  2. Each fact is verified probabilistically in the image
  3. Geometric mean of all verification scores = final score
- **Type**: 🔍 VLM-based fact extraction + verification
- **Range**: 0-100 (higher = better alignment)
- **Why it's the North Star**: Most comprehensive measure of whether the image matches the prompt

---

### 🎯 ALIGNMENT METRICS
All metrics measure how well the image matches your text prompt:

#### **VQAScore** 🤖 Model
- **Method**: ViLT (Vision-and-Language Transformer) model
- **How it works**: Asks questions about the image based on your prompt and evaluates answers
- **Strength**: Direct question-answering approach
- **Range**: 0-100

#### **CLIPScore** 🤖 Model
- **Method**: OpenAI CLIP embeddings
- **How it works**: Calculates cosine similarity between text and image embeddings
- **Strength**: Industry-standard, widely validated
- **Range**: 0-100 (normalized)

#### **CMMD** (Cross-Modal Matching Distance) 🤖 Model
- **Method**: CLIP-based cross-modal distance
- **How it works**: Measures distance between text and image in CLIP's shared embedding space
- **Strength**: Captures semantic alignment
- **Range**: 0-100 (inverted - lower distance = higher score)

#### **AHEaD** (Alignment Head) 🤖 Model
- **Method**: CLIP attention mechanism
- **How it works**: Analyzes where the model "looks" when matching text to image
- **Strength**: Attention-based alignment verification
- **Range**: 0-100

#### **PickScore** 🤖 Model
- **Method**: CLIP + aesthetic preferences
- **How it works**: Estimates human preference for the image given the prompt
- **Strength**: Predicts subjective human judgment
- **Range**: 0-100

---

### 🖼️ IMAGE QUALITY METRICS
These metrics evaluate technical quality **independent of your prompt**:

#### **BRISQUE** (Blind/Referenceless Image Spatial Quality Evaluator) 📐 Code
- **Method**: Statistical natural scene analysis
- **How it works**: Analyzes spatial statistics without needing a reference image
- **Strength**: Fast, no ML model needed, good for detecting distortions
- **Range**: 0-100 (lower raw BRISQUE = higher quality, normalized in report)
- **What it catches**: Blur, noise, compression artifacts

#### **NIQE** (Natural Image Quality Evaluator) 📐 Code
- **Method**: Natural scene statistics model
- **How it works**: Compares image statistics to pristine natural image database
- **Strength**: No reference needed, purely algorithmic
- **Range**: 0-100 (lower raw NIQE = higher quality, normalized in report)
- **What it catches**: Unnaturalness, distortion, lack of sharpness

#### **CLIP-IQA** 🤖 Model
- **Method**: CLIP embeddings for quality assessment
- **How it works**: Uses CLIP to estimate aesthetic and technical quality
- **Strength**: Learned quality assessment from data
- **Range**: 0-100
- **What it catches**: Overall visual appeal and quality

---

### 🛡️ SAFETY METRICS
All use GPT-4o to analyze potential ethical and safety concerns:

#### **Toxicity Safety** 🔍 VLM
- **Checks for**: Hate speech, violence, NSFW content, disturbing imagery
- **Method**: GPT-4o visual inspection
- **Range**: 0-100 (higher = safer)

#### **Fairness** 🔍 VLM
- **Checks for**: Stereotypes, bias, marginalization, cultural insensitivity
- **Method**: GPT-4o bias detection
- **Range**: 0-100 (higher = fairer)

#### **Privacy Safety** 🔍 VLM
- **Checks for**: Identifiable faces, personal info, license plates, private documents
- **Method**: GPT-4o privacy analysis
- **Range**: 0-100 (higher = more private)

---

## 💡 QUALITATIVE ASSESSMENT

The "Qualitative Assessment" section contains GPT-4o's subjective, human-like evaluation in natural language. This includes:
- Overall impressions
- Specific observations about composition, lighting, aesthetics
- Analysis of how well elements match the prompt
- Identification of any notable issues or strengths

**Note**: This is **subjective** and complements the objective metrics above.

---

## 🔍 How to Interpret Your Report

### Good Scores Generally Mean:
- **Soft-TIFA GM 80+**: Excellent prompt alignment
- **VQAScore/CLIPScore 70+**: Strong text-image correspondence
- **BRISQUE/NIQE 80+**: High technical quality
- **Safety Metrics 95+**: No significant concerns

### Red Flags to Watch For:
- **Large gap between metrics**: E.g., high CLIPScore but low VQAScore may indicate superficial matching
- **Low Soft-TIFA with specific atoms failing**: Check which facts weren't captured
- **Safety scores <90**: Review the specific issues identified
- **All metrics high but low qualitative score**: May indicate the objective metrics miss subjective quality issues

---

## 🎯 Which Metrics Should You Trust Most?

**For Alignment:**
1. **Soft-TIFA GM** (most comprehensive)
2. **VQAScore** (direct verification)
3. **CLIPScore** (industry standard)

**For Technical Quality:**
1. **Qualitative Assessment** (human-like judgment)
2. **CLIP-IQA** (learned preferences)
3. **BRISQUE/NIQE** (specific technical issues)

**For Safety:**
- All three safety metrics are equally important - review specific issues identified

---

## 📊 Performance Metrics Explained

- **Time to First Token**: How long until GPT-4o starts responding
- **VLM Evaluation**: Total time for GPT-4o qualitative assessment
- **Soft-TIFA Calculation**: Time to extract and verify facts (slowest, most thorough)
- **Total Processing**: End-to-end evaluation time

---

## 🚀 Tips for Better Results

1. **Be specific in prompts**: "A woman with blue hair" → Soft-TIFA can verify specific facts
2. **Check atoms that failed**: If Soft-TIFA is low, see which specific facts weren't captured
3. **Compare metrics**: If alignment is high but quality is low, the model understood but executed poorly
4. **Review safety issues**: Even minor fairness concerns are worth noting for production use

---

**Need More Help?** Check the README.md for full documentation or review the source code in `src/app.py`.
