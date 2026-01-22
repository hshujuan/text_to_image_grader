# Text-to-Image Grader Redesign Architecture

## 📐 Design Philosophy

The redesigned grader follows a **hierarchical metric structure** with clear separation between:
1. **North Star Metric** (Primary quality indicator)
2. **Supporting Metrics** (Multi-dimensional quality assessment)

---

## ⭐ North Star Metric: Soft-TIFA GM

### What is Soft-TIFA?
**Soft-TIFA (Soft Text-Image Faithfulness through Atomic evaluation)** measures how well an image satisfies atomic visual criteria extracted from the prompt.

### Why Geometric Mean (GM)?
- **Compositional AND Logic:** All criteria must be satisfied (not just averaged)
- **Penalizes Missing Elements:** A single 0 score results in 0 overall score
- **Balanced Assessment:** Better reflects human judgment on completeness

### Methodology:
```
1. Extract 5-8 atomic facts from prompt
   Examples: "a red car", "on a highway", "sunny weather"

2. Score each atom probabilistically (0.0 to 1.0)
   VLM evaluates: Is this criterion met in the image?

3. Calculate Geometric Mean:
   Soft-TIFA GM = (score₁ × score₂ × ... × scoreₙ)^(1/n) × 100
```

### Example:
```
Prompt: "A red car on a highway with mountains in background"

Atoms:
- "A car is present" → 1.0 ✓
- "The car is red" → 0.9 ✓ (slightly orange-red)
- "Car is on a highway" → 0.8 ✓ (looks like highway)
- "Mountains in background" → 0.0 ✗ (no mountains visible)

Soft-TIFA GM = (1.0 × 0.9 × 0.8 × 0.0)^(1/4) × 100 = 0/100
```

**Result:** Low score because mountains are missing (compositional AND logic).

---

## 📊 Supporting Metrics

### A. Image-Only Quality Metrics
**Purpose:** Assess technical quality independent of prompt

| Metric | What It Measures |
|--------|------------------|
| **BRISQUE** | Blind spatial quality (blur, noise, compression artifacts) |
| **NIQE** | Natural image statistics (deviation from natural distributions) |
| **CLIP-IQA** | CLIP-based perceptual quality |

**Implementation:** `src/metrics/image_quality.py` - Laplacian-based quality estimation

---

### B. Alignment Metrics
**Purpose:** Measure text-image correspondence

#### Model-Based (Fast)
| Metric | What It Measures | Best For |
|--------|------------------|----------|
| **CLIPScore** | Global semantic alignment (embedding cosine similarity) | Overall intent match |
| **VQAScore** | Visual QA-based verification | Factual correctness |
| **AHEaD** | CLIP attention-based alignment | Fine-grained matching |
| **PickScore** | Human preference estimation | Subjective quality |

#### VLM-Based (GPT-4o)
| Metric | What It Measures | Best For |
|--------|------------------|----------|
| **TIFA** | QA pair verification | Factual accuracy |
| **DSG** | Davidsonian semantic primitives | Structured verification |
| **PSG** | Scene graph structure | Object relationships |
| **VPEval** | Visual programming evaluation | Compositional reasoning |

**Implementation:** `src/metrics/alignment.py` - CLIP/ViLT models + GPT-4o VLM

---

### C. Safety Metrics (T2ISafety Framework)
**Purpose:** Responsible AI evaluation

| Metric | What It Measures |
|--------|------------------|
| **Toxicity** | Harmful content (hate speech, violence, NSFW) |
| **Fairness** | Bias and stereotyping |
| **Privacy** | Identifiable information |

**Implementation:** `src/metrics/safety.py` - GPT-4o VLM evaluation

---

## 🎯 Evaluation Workflow

### Single Image Evaluation
```
1. Calculate Soft-TIFA GM (North Star)
   - Extract atomic criteria from prompt
   - Evaluate each atom probabilistically
   - Calculate geometric mean
   ↓
2. Run T2ISafety evaluation
   ↓
3. Calculate Image Quality metrics (BRISQUE, NIQE, CLIP-IQA)
   ↓
4. Calculate Model-based Alignment metrics (CLIPScore, VQAScore, AHEaD, PickScore)
   ↓
5. Calculate VLM-based Alignment metrics (TIFA, DSG, PSG, VPEval)
   ↓
6. Run Expert VLM evaluation (GPT-4o qualitative assessment)
   ↓
6. Generate comprehensive report
```

### Batch Evaluation
```
For each image:
1. Extract atoms → Evaluate atoms → Calculate Soft-TIFA GM
2. Record results with all metrics

Summary:
- Average Soft-TIFA GM (primary metric)
- Per-image breakdown
```

---

## 📈 Report Structure

The report is organized in this order:

1. **⭐ North Star Metric** - Soft-TIFA GM score prominently displayed
2. **🔬 Soft-TIFA Atomic Fact Verification** - Detailed breakdown of each criterion
3. **💡 Expert VLM Evaluation** - GPT-4o subjective assessment
4. **🎯 Alignment Metrics** - Model-based (CLIPScore, VQAScore, AHEaD, PickScore) + VLM-based (TIFA, DSG, PSG, VPEval)
5. **🖼️ Image Quality Metrics** - BRISQUE, NIQE, CLIP-IQA
6. **🛡️ Safety Metrics** - Toxicity, Fairness, Privacy
7. **📊 Overall Summary** - Category averages

*Performance metrics are displayed separately under the generated image.*

---

## 🔧 Implementation Details

### Project Structure
```
src/
├── app.py              # Main Gradio application
├── openai_service.py   # DALL-E 3 image generation
└── metrics/            # Shared metrics module
    ├── __init__.py     # Module exports
    ├── utils.py        # Utilities (pil_to_base64, model loaders)
    ├── soft_tifa.py    # North Star metric
    ├── image_quality.py # BRISQUE, NIQE, CLIP-IQA
    ├── alignment.py    # CLIPScore, VQAScore, AHEaD, PickScore, TIFA, DSG, PSG, VPEval
    └── safety.py       # T2ISafety evaluation
```

### Current Implementation
- **Soft-TIFA GM**: VLM-based atom extraction and verification (GPT-4o)
- **CLIPScore**: Real CLIP embeddings cosine similarity
- **VQAScore**: Real ViLT model question answering
- **AHEaD**: CLIP attention-based alignment metric
- **TIFA/DSG/PSG/VPEval**: VLM-based alignment metrics (GPT-4o)
- **Image Quality**: Laplacian-based estimation (OpenCV)
- **Safety**: GPT-4o VLM evaluation

---

## 🎓 When to Use Each Metric

### Soft-TIFA GM (Always Report First)
- **Use for:** Primary quality indicator, benchmark comparisons
- **Strengths:** Fine-grained correctness, compositional logic

### Image Quality Metrics
- **Use for:** Technical assessment independent of prompt
- **Best for:** Debugging generation quality, comparing models

### Alignment Metrics
- **CLIPScore:** Quick global semantic assessment
- **VQAScore:** Factual verification via Q&A
- **AHEaD:** Fine-grained CLIP-based alignment
- **TIFA/DSG/PSG/VPEval:** VLM-based structured verification

### Safety Metrics
- **Use for:** Responsible AI evaluation
- **Best for:** Production deployment, bias detection

---

## 📊 Metric Selection Guide

| Task | Primary Metric | Supporting Metrics |
|------|----------------|-------------------|
| **Benchmark T2I Models** | Soft-TIFA GM | CLIPScore, Image Quality |
| **Debug Generation Quality** | Image Quality | CLIP-IQA, VLM Quality Estimate |
| **Verify Prompt Adherence** | Soft-TIFA GM | VQAScore, Attribute Accuracy |
| **Evaluate Complex Prompts** | Soft-TIFA GM | VLM-as-a-Judge (Reasoning) |
| **Compare Model Outputs** | Soft-TIFA GM | All Alignment + Quality |
| **Style Transfer** | LPIPS | Image Quality |

---

## 🚀 Quick Start

### 1. Run the Application
```powershell
# Activate the conda environment
conda activate t2i_grader

# Or use the Python executable directly
C:\Users\hshuj\.conda\envs\t2i_grader\python.exe src/app.py
```

### 2. Single Image Evaluation
- Upload image + enter prompt → Click "Grade Image Quality"
- Get comprehensive report with all metrics

### 3. Batch Benchmarking
Upload CSV with `prompt` column (optional `image_path`):
```csv
prompt
"A red apple on a wooden table"
"A steampunk workshop with brass gears"
```

---

## 📚 References

**Soft-TIFA:**
- Hu et al. "TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering"

**Image Quality:**
- BRISQUE: Mittal et al. "No-Reference Image Quality Assessment in the Spatial Domain"
- NIQE: Mittal et al. "Making a Completely Blind Image Quality Analyzer"

**Alignment:**
- CLIPScore: Hessel et al. "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
- VQAScore: Lin et al. "Evaluating Text-to-Visual Generation with Image-to-Text Generation"

**Safety:**
- T2ISafety: Framework for responsible AI evaluation in text-to-image generation

---

## ✅ Summary

**Key Design Decisions:**
1. ⭐ **North Star First:** Soft-TIFA GM reported prominently
2. 📊 **Hierarchical Structure:** North Star → VLM Evaluation → Alignment → Quality → Safety
3. 🔧 **Modular Implementation:** Shared `metrics/` module for all metric calculations
4. 🎯 **Use-Case Driven:** Different metrics for different evaluation needs

**Architecture:**
- Single entry point: `app.py`
- Shared metrics module: `metrics/`
- Performance metrics displayed under the image
- Comprehensive report on the right side
