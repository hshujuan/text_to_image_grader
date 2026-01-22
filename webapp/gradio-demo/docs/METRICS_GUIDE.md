# 📊 Metrics Guide: Understanding Your Image Quality Report

This guide explains the metrics and report structure used to evaluate your generated images.

---

## 📈 Report Structure

Your report is organized in this order:

1. **⭐ North Star Metric** - Primary quality indicator (Soft-TIFA GM)
2. **🔬 Soft-TIFA Atomic Fact Verification** - Detailed breakdown of extracted criteria
3. **💡 Expert VLM Evaluation** - GPT-4o subjective quality assessment
4. **🎯 Alignment Metrics** - Text-image correspondence scores
5. **🖼️ Image Quality Metrics** - Technical quality assessment
6. **🛡️ Safety Metrics** - Responsible AI evaluation
7. **📊 Overall Summary** - Quick category scores overview

*Performance metrics are shown separately under the generated image.*

---

## ⭐ NORTH STAR METRIC

### **Soft-TIFA GM** (Geometric Mean)
- **What it measures**: Overall text-image alignment accuracy
- **How it works**: 
  1. GPT-4o extracts atomic facts from your prompt (e.g., "steampunk workshop", "brass gears", "mechanical owl")
  2. Each fact is verified probabilistically in the image (0.0 to 1.0)
  3. Geometric mean of all verification scores × 100 = final score
- **Range**: 0-100 (higher = better alignment)
- **Good score**: 80+
- **Why it's the North Star**: Uses compositional AND logic - all facts must be present for a high score

---

## 💡 EXPERT VLM EVALUATION

GPT-4o provides subjective, human-like assessment including:
- **Image Quality**: Clarity, resolution, artifacts, composition, lighting
- **Prompt Adherence**: How well elements match the description
- **Overall Summary**: Executive summary of generation quality

This complements the objective metrics with nuanced analysis.

---

## 🎯 ALIGNMENT METRICS

These measure how well the image matches your text prompt:

### Model-Based (Fast)
| Metric | Method | Good Score |
|--------|--------|------------|
| **VQAScore** | ViLT visual question answering | 70+ |
| **CLIPScore** | CLIP embedding cosine similarity | 70+ |
| **AHEaD** | CLIP attention-based alignment | 60+ |
| **PickScore** | Human preference estimation | 70+ |

### VLM-Based (GPT-4o)
| Metric | Method | Good Score |
|--------|--------|------------|
| **TIFA** | Question-answer pair verification | 70+ |
| **DSG** | Davidsonian Scene Graph decomposition | 70+ |
| **PSG** | Panoptic Scene Graph evaluation | 70+ |
| **VPEval** | Visual Programming evaluation | 70+ |

---

## 🖼️ IMAGE QUALITY METRICS

These evaluate technical quality **independent of your prompt**:

| Metric | Method | Good Score | Detects |
|--------|--------|------------|---------|
| **BRISQUE** | Spatial quality analysis | 80+ | Blur, noise, compression |
| **NIQE** | Natural scene statistics | 80+ | Unnaturalness, distortion |
| **CLIP-IQA** | CLIP quality assessment | 70+ | Overall visual appeal |

---

## 🛡️ SAFETY METRICS (T2ISafety Framework)

All use GPT-4o to analyze potential ethical and safety concerns:

| Metric | Checks For | Good Score |
|--------|------------|------------|
| **Toxicity** | Hate speech, violence, NSFW, disturbing imagery | 95+ |
| **Fairness** | Stereotypes, bias, marginalization, cultural insensitivity | 95+ |
| **Privacy** | Identifiable faces, personal info, private documents | 95+ |

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

## � Implementation Details

This section documents how each metric is calculated - whether using external open-source packages or custom implementations.

### Image Quality Metrics

| Metric | Primary Implementation | Package | Fallback |
|--------|----------------------|---------|----------|
| **BRISQUE** | ✅ External Package | `piq` library | Custom (OpenCV gradient analysis) |
| **NIQE** | ✅ External Package | `piq` library | Custom (entropy + edge density) |
| **CLIP-IQA** | ⚠️ Fallback Only | `pyiqa` (not installed)* | Custom (sharpness + contrast + color) |

*`pyiqa` has dependency conflicts with modern `transformers` versions, so the fallback is used.

### Alignment Metrics

| Metric | Primary Implementation | Package | Fallback |
|--------|----------------------|---------|----------|
| **CLIPScore** | ✅ External Package | `torchmetrics.multimodal` | Custom (OpenAI CLIP embeddings) |
| **PickScore** | ✅ External Package | HuggingFace `transformers` (`yuvalkirstain/PickScore_v1`) | CLIP + aesthetics proxy |
| **VQAScore** | ✅ Custom | ViLT model via `transformers` | N/A |
| **AHEaD** | ✅ Custom | OpenAI CLIP attention patterns | N/A |
| **TIFA** | ✅ Custom | Azure OpenAI GPT-4o | N/A |
| **DSG** | ✅ Custom | Azure OpenAI GPT-4o | N/A |
| **PSG** | ✅ Custom | Azure OpenAI GPT-4o | N/A |
| **VPEval** | ✅ Custom | Azure OpenAI GPT-4o | N/A |

### North Star Metric

| Metric | Implementation | Description |
|--------|---------------|-------------|
| **Soft-TIFA GM** | ✅ Custom | GPT-4o extracts atomic facts, then verifies each via VQA. Geometric mean of verification scores. |

### Safety Metrics

| Metric | Implementation | Description |
|--------|---------------|-------------|
| **Toxicity** | ✅ Custom | Azure OpenAI GPT-4o content analysis |
| **Fairness** | ✅ Custom | Azure OpenAI GPT-4o bias detection |
| **Privacy** | ✅ Custom | Azure OpenAI GPT-4o privacy check |

### Package Dependencies

```
# Image Quality Assessment
piq                    # BRISQUE, NIQE (official implementations)
torchmetrics[multimodal]  # CLIPScore from TorchMetrics

# Models
transformers           # PickScore (HuggingFace), ViLT (VQA)
clip (OpenAI)          # CLIP embeddings for AHEaD, fallback CLIPScore

# VLM-Based Metrics
azure-ai-inference     # GPT-4o for TIFA, DSG, PSG, VPEval, Safety, Soft-TIFA
```

### Why External Packages?

| Package | Reason |
|---------|--------|
| `piq` | Battle-tested, GPU-accelerated, matches academic implementations |
| `torchmetrics` | Standard ML metrics library, well-maintained, consistent API |
| HuggingFace `transformers` | Access to pre-trained models (PickScore, ViLT) |

### Fallback Strategy

All metrics have graceful degradation:
1. **Try external package** (most accurate)
2. **Fall back to custom implementation** (if package unavailable)
3. **Return 0.0 with warning** (if both fail)

This ensures the app runs even if some packages aren't installed.

---

## �🚀 Tips for Better Results

1. **Be specific in prompts**: "A woman with blue hair" → Soft-TIFA can verify specific facts
2. **Check atoms that failed**: If Soft-TIFA is low, see which specific facts weren't captured
3. **Compare metrics**: If alignment is high but quality is low, the model understood but executed poorly
4. **Review safety issues**: Even minor fairness concerns are worth noting for production use

---

**Need More Help?** Check the README.md for full documentation or review the source code in `src/app.py`.
