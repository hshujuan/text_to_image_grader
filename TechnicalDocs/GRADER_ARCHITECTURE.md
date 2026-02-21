# Text-to-Image Grader Redesign Architecture

## 📐 Design Philosophy

The redesigned grader follows a **hierarchical metric structure** with clear separation between:
1. **North Star Metrics** (Three complementary faithfulness indicators)
2. **Supporting Metrics** (Multi-dimensional quality assessment)

---

## ⭐ North Star Metrics

Three complementary faithfulness metrics, each representing a different evaluation paradigm:

### 1. Soft-TIFA GM — The Probabilistic Fact-Checker *(Meta, GenEval 2)*

**Soft-TIFA (Soft Text-Image Faithfulness through Atomic evaluation)** measures how well an image satisfies atomic visual criteria extracted from the prompt.

- **Methodology:** Extract atomic facts → score each via VLM token log-probabilities (0.0–1.0) → geometric mean
- **Compositional AND Logic:** All criteria must be satisfied (not just averaged)
- **Penalizes Missing Elements:** A single 0 score results in 0 overall score
- **Range:** 0-100 | **Good score:** 80+

```
Soft-TIFA GM = (score₁ × score₂ × ... × scoreₙ)^(1/n) × 100
```

### 2. DSG — The Structural Logician *(Google, ICLR 2024)*

**DSG (Davidsonian Scene Graph)** evaluates logical faithfulness with dependency validity.

- **Methodology:** 3-stage LLM pipeline (tuples → questions → dependency DAG) → binary yes/no VQA → dependency filtering
- **Key Strength:** If an object is absent, its attributes/relations are automatically zeroed out (no false credit)
- **Range:** 0-100 | **Good score:** 70+

### 3. PSG — The Visual Surveyor *(ByteDance, ICCV 2025)*

**PSG (Panoptic Scene Graph)** evaluates structural scene-graph alignment.

- **Methodology:** Build scene graphs from both prompt and image → match objects, attributes, relations → F1 score
- **Key Strength:** Evaluates objects, attributes, and relations as separate dimensions — penalizes both extras and omissions
- **Range:** 0-100 | **Good score:** 70+

**Why three?** Each metric trusts a different signal: Soft-TIFA trusts token probabilities, DSG trusts logical structure, PSG trusts visual parsing.

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

*Note: DSG and PSG are North Star metrics. TIFA and VPEval were removed — Soft-TIFA GM already covers probabilistic fact-checking, and DSG/PSG provide stronger structural signals.*

**Implementation:** `src/metrics/alignment.py` - CLIP/ViLT models

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
1. Calculate North Star Metrics (Soft-TIFA GM, DSG, PSG)
   - Soft-TIFA: Extract atomic criteria → probabilistic verification → geometric mean
   - DSG: Tuple extraction → dependency DAG → VQA verification
   - PSG: Scene graph extraction → structural matching → F1 score
   ↓
2. Run T2ISafety evaluation
   ↓
3. Calculate Image Quality metrics (BRISQUE, NIQE, CLIP-IQA)
   ↓
4. Calculate Model-based Alignment metrics (CLIPScore, VQAScore, AHEaD, PickScore)
   ↓
5. Run Expert VLM evaluation (GPT-4o qualitative assessment)
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

1. **⭐ North Star Metrics** - Soft-TIFA GM, DSG, and PSG scores prominently displayed
2. **🔬 Soft-TIFA Atomic Fact Verification** - Detailed breakdown of each criterion
3. **💡 Expert VLM Evaluation** - GPT-4o subjective assessment
4. **🎯 Alignment Metrics** - Model-based (CLIPScore, VQAScore, AHEaD, PickScore)
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
    ├── alignment.py    # CLIPScore, VQAScore, AHEaD, PickScore, DSG, PSG
    └── safety.py       # T2ISafety evaluation
```

### Current Implementation
- **Soft-TIFA GM**: VLM-based atom extraction and verification (GPT-4o)
- **CLIPScore**: Real CLIP embeddings cosine similarity
- **VQAScore**: Real ViLT model question answering
- **AHEaD**: CLIP attention-based alignment metric
- **DSG/PSG**: VLM-based alignment metrics (GPT-4o)
- **Image Quality**: Laplacian-based estimation (OpenCV)
- **Safety**: GPT-4o VLM evaluation

---

## 🎓 When to Use Each Metric

### North Star Metrics (Always Report First)
- **Soft-TIFA GM:** Primary quality indicator — fine-grained probabilistic correctness
- **DSG:** Structural logical verification with dependency filtering
- **PSG:** Scene graph alignment with object/attribute/relation matching

### Image Quality Metrics
- **Use for:** Technical assessment independent of prompt
- **Best for:** Debugging generation quality, comparing models

### Alignment Metrics
- **CLIPScore:** Quick global semantic assessment
- **VQAScore:** Factual verification via Q&A
- **AHEaD:** Fine-grained CLIP-based alignment
- **VQAScore:** Direct factual verification

### Safety Metrics
- **Use for:** Responsible AI evaluation
- **Best for:** Production deployment, bias detection

---

## 📊 Metric Selection Guide

| Task | Primary Metric | Supporting Metrics |
|------|----------------|-------------------|
| **Benchmark T2I Models** | Soft-TIFA GM, DSG, PSG | CLIPScore, Image Quality |
| **Debug Generation Quality** | Image Quality | CLIP-IQA, VLM Quality Estimate |
| **Verify Prompt Adherence** | Soft-TIFA GM, DSG | VQAScore, Attribute Accuracy |
| **Evaluate Complex Prompts** | Soft-TIFA GM, PSG | VLM-as-a-Judge (Reasoning) |
| **Compare Model Outputs** | All 3 North Stars | All Alignment + Quality |
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
1. ⭐ **Three North Stars:** Soft-TIFA GM, DSG, and PSG reported prominently — each trusts a different signal
2. 📊 **Hierarchical Structure:** North Stars → VLM Evaluation → Alignment → Quality → Safety
3. 🔧 **Modular Implementation:** Shared `metrics/` module for all metric calculations
4. 🎯 **Use-Case Driven:** Different metrics for different evaluation needs

**Architecture:**
- Root launcher: `run.py`
- Single entry point: `webapp/gradio-demo/src/app.py`
- Shared metrics module: `webapp/gradio-demo/src/metrics/`
- Performance metrics displayed under the image
- Comprehensive report on the right side
