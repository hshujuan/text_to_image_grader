# Azure DALL-E 3 Text-to-Image Generator & Grader

A comprehensive Gradio web app for generating and evaluating text-to-image outputs using Azure OpenAI DALL-E 3 and GPT-4o, with real metric calculations — not LLM estimates.

## Features

✨ **Image Generation**: DALL-E 3 powered high-quality image generation  
🎯 **Comprehensive Evaluation**: 15+ metrics across alignment, image quality, and safety  
⭐ **Three North Star Metrics**: Soft-TIFA GM, DSG, and PSG for multi-paradigm faithfulness  
📊 **Batch Processing**: CSV-based batch generation and grading with smart caching  
⚡ **Performance Tracking**: Time-to-first-token and detailed timing metrics  
🔬 **Real Metrics**: Actual model-based calculations (CLIP, VQA, etc.)  
🧪 **GenEval2 Benchmark**: Built-in prompts from the [GenEval2](https://github.com/facebookresearch/GenEval2) benchmark with pre-defined VQA atoms

📖 **[Read the Metrics Guide](TechnicalDocs/METRICS_GUIDE.md)** to understand what each metric measures and how to interpret your results.

## North Star Metric Architecture

The evaluation system uses **three complementary North Star metrics**, each representing a different evaluation paradigm:

### ⭐ North Star 1: Soft-TIFA GM — The Probabilistic Fact-Checker *(Meta, GenEval 2)*
- **Atomic fact extraction** + probabilistic verification via VLM token log-probabilities  
- **Geometric Mean**: Penalizes any single failed atom — compositional AND logic  
- **Primary quality indicator**: Main score for text-image alignment

### ⭐ North Star 2: DSG — The Structural Logician *(Google, ICLR 2024)*
- **3-stage LLM pipeline**: Tuples → questions → dependency DAG → binary VQA  
- **Dependency filtering**: If an object is absent, its attributes/relations auto-zero  
- Captures logical faithfulness with structured verification

### ⭐ North Star 3: PSG — The Visual Surveyor *(ByteDance, ICCV 2025)*
- **Scene graph matching**: Builds graphs from prompt and image, matches objects/attributes/relations  
- **Structural alignment**: Evaluates objects, attributes, and relations as separate dimensions  
- Penalizes both extras and omissions

### 🎯 Supporting Alignment Metrics

#### Model-Based (Fast, Local)
- **VQAScore**: Real VQA model (ViLT) for visual question answering (✅ model-based)
- **CLIPScore**: Real CLIP embeddings cosine similarity (✅ model-based)
- **AHEaD**: Alignment Head score using CLIP attention (✅ model-based)
- **PickScore**: Human preference proxy using CLIP + aesthetics

#### VLM-Based (GPT-4o)
- **TIFA**: Text-to-Image Faithfulness via QA pair verification
- **VPEval**: Visual Programming evaluation

### 🖼️ Technical Image Quality Metrics  
These metrics evaluate image quality independent of the text prompt:
- **BRISQUE**: Blind/Referenceless Image Spatial Quality Evaluator  
- **NIQE**: Natural Image Quality Evaluator  
- **CLIP-IQA**: CLIP-based Image Quality Assessment

### 🛡️ T2ISafety Framework
- **Toxicity Safety**: Harmful content detection (hate speech, violence, NSFW)
- **Fairness**: Bias and stereotyping assessment  
- **Privacy Safety**: Privacy concerns (identifiable data, personal info)

## Project Structure

```
text_to_image_grader/
├── run.py                     # Root launcher (python run.py)
├── README.md
├── webapp/
│   └── gradio-demo/
│       ├── src/
│       │   ├── app.py             # Main Gradio app with comprehensive grading system
│       │   ├── openai_service.py  # Azure OpenAI DALL-E 3 image generation
│       │   └── metrics/           # Metrics module
│       │       ├── __init__.py    # Module exports
│       │       ├── utils.py       # Shared utilities (pil_to_base64, model loaders)
│       │       ├── soft_tifa.py   # North Star: Soft-TIFA GM implementation
│       │       ├── alignment.py   # CLIPScore, VQAScore, AHEaD, PickScore, TIFA, DSG, PSG, VPEval
│       │       ├── image_quality.py # BRISQUE, NIQE, CLIP-IQA
│       │       └── safety.py      # T2ISafety evaluation
│       ├── tests/                 # Test suite
│       │   └── test_vqa_comparison.py
│       ├── test_data/             # Test datasets
│       │   ├── T2I_tests.csv      # Full test prompts for batch evaluation
│       │   └── T2I_tests_small.csv # Smaller test set for quick testing
│       ├── batch_generated_images/ # Cached generated images (by prompt hash)
│       ├── docs/                  # Human evaluation guidelines
│       ├── requirements.txt       # Python dependencies
│       ├── .env                   # Environment variables (not committed)
│       └── .env.example           # Example environment file (no secrets)
├── lib/                           # Reference implementations and data
│   ├── DSG/                       # Davidsonian Scene Graph (Google)
│   └── geneval2/                  # GenEval2 benchmark data (Meta)
├── TechnicalDocs/                 # Architecture and design documentation
│   ├── GRADER_ARCHITECTURE.md     # Metric architecture design
│   ├── METRICS_GUIDE.md           # Comprehensive metrics guide
│   ├── MetricsSystem.md           # Metrics system overview
│   └── DocumentationIndex.md     # Documentation index
└── apple/                         # Python virtual environment (not committed)
```

## Installation

1. Clone the repository and install dependencies:

```powershell
git clone <repository-url>
cd text_to_image_grader/webapp/gradio-demo
python -m pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:

```powershell
copy .env.example .env
# Edit .env and set your API keys
```

## Running the Application

From the project root:

```powershell
python run.py
```

Or directly:

```powershell
python webapp/gradio-demo/src/app.py
```

This starts a local server at `http://localhost:7860`.

## Usage

The application has three tabs:

### 🖼️ Tab 1: Generate & Grade
1. **Enter a prompt** or click one of the curated sample prompts:
   - 🟢 **GenEval2 Easy**: "a elephant and a purple kangaroo"
   - 🟡 **GenEval2 Spatial**: "a candle, and a blue truck in front of a cookie"
   - 🔴 **GenEval2 Hard**: "seven wooden rabbits, and three green horses in front of a striped chair"
   - 🟣 **Complex (free-text)**: "A small child holding a glowing lantern while standing next to a golden retriever in a snowy forest at dusk"

   *The first three prompts come from the [GenEval2](https://github.com/facebookresearch/GenEval2) benchmark and include pre-defined VQA atoms for precise evaluation.*

2. **Generate Image**: Click "🚀 Generate Image" to create an image using Azure DALL-E 3

3. **Grade Quality**: Click "📊 Grade Image Quality" to run comprehensive evaluation
   - Progress bar shows 6 steps: Soft-TIFA GM → T2ISafety → Image Quality → Model Alignment → VLM Alignment → Expert Evaluation
   - Report order: North Star → Soft-TIFA Details → Expert VLM Evaluation → Alignment → Image Quality → Safety → Overall Summary
   - Performance metrics displayed under the generated image

### 📊 Tab 2: Batch Scoring
Evaluate multiple images at once with three flexible modes:

| Mode | CSV Columns | Description |
|------|-------------|-------------|
| **Mode 1: Auto-Generate** | `prompt` (+ `category`) | Generates images with DALL-E 3 |
| **Mode 2: Grade Existing** | `prompt`, `image_path` (+ `category`) | Loads images from specified paths |
| **Mode 3: Hybrid** | Same as Mode 2 + checkbox | Generates missing images automatically |

*Note: `category` column is optional for all modes*

**Features:**
- **Smart Caching**: Images cached by prompt hash — re-running uses cache (FREE!)
- **Generate Missing Images**: Check "🎨 Generate Missing Images (DALL-E 3)" to auto-generate any missing images
- **Pass/Fail Tracking**: Soft-TIFA Score ≥ 80 = Pass, with overall pass rate summary
- **Downloadable Results**: CSV output with all metrics, scores, and pass/fail status

**Example CSV (Mode 1):**
```csv
prompt,category
"A red cat on a blue sofa",simple
"A woman with blonde hair",portrait
```

**Example CSV (Mode 2/3):**
```csv
prompt,image_path,category
"A red cat on a blue sofa",./images/image1.png,simple
"A woman with blonde hair",./images/image2.png,portrait
```

### 📖 Tab 3: Metrics Guide
- **In-app documentation**: Comprehensive guide explaining all metrics
- **Metric types**: Learn the difference between 🤖 Model, 📐 Code, and 🔍 VLM-based metrics
- **Interpretation help**: What good scores look like and how to debug issues

## Environment Variables

The app uses a `.env` file for configuration:

### Image Generation (DALL-E 3)
```
AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.cognitiveservices.azure.com/
OPENAI_API_VERSION=2024-04-01-preview
DEPLOYMENT_NAME=dall-e-3
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
```

### Grading System (GPT-4o)
```
AZURE_OPENAI_GRADING_ENDPOINT=https://<your-endpoint>.openai.azure.com/openai/v1/
AZURE_OPENAI_GRADING_KEY=your-grading-api-key-here
GRADING_DEPLOYMENT_NAME=gpt-4o
GRADING_API_VERSION=2024-02-15-preview
```

**Note**: You can use the same Azure OpenAI resource for both generation and grading, or separate resources.

## Technical Documentation

Detailed architecture and design docs are in the [TechnicalDocs/](TechnicalDocs/) folder:
- [GRADER_ARCHITECTURE.md](TechnicalDocs/GRADER_ARCHITECTURE.md) — Hierarchical metric design
- [METRICS_GUIDE.md](TechnicalDocs/METRICS_GUIDE.md) — Comprehensive metrics reference
- [MetricsSystem.md](TechnicalDocs/MetricsSystem.md) — System overview
- [DocumentationIndex.md](TechnicalDocs/DocumentationIndex.md) — Full documentation index

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.

---

**Security Note:**
- Do not commit your `.env` file with real API keys to version control. Use `.env.example` for sharing variable names only.