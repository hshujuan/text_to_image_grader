# Azure DALL-E 3 Text-to-Image Generator & Grader

This project provides a comprehensive Gradio application for generating and evaluating text-to-image outputs using Azure OpenAI DALL-E 3 and GPT-4o, with real metric calculations.

## Features

✨ **Image Generation**: DALL-E 3 powered high-quality image generation  
🎯 **Comprehensive Evaluation**: Multi-metric quality assessment with North Star architecture  
📊 **Batch Processing**: CSV-based batch generation and grading with smart caching  
⚡ **Performance Tracking**: Time-to-first-token and detailed timing metrics  
🔬 **Real Metrics**: Actual model-based calculations (CLIP, VQA, etc.) not LLM estimates

## North Star Metric Architecture

Our evaluation system uses **Soft-TIFA Geometric Mean** as the primary quality indicator, supported by:

### ⭐ North Star: Soft-TIFA GM
- **True Implementation**: Atomic fact extraction + probabilistic verification  
- **Geometric Mean Calculation**: Actual methodology, not estimated  
- **Primary Quality Indicator**: Main score for text-image alignment

### 🎯 Supporting Alignment Metrics
- **VQAScore**: Real VQA model (ViLT) for visual question answering (✅ model-based)
- **CLIPScore**: Real CLIP embeddings cosine similarity (✅ model-based)
- **CMMD**: Cross-Modal Matching Distance using CLIP (✅ model-based)  
- **AHEaD**: Alignment Head score using CLIP attention (✅ model-based)
- **PickScore**: Human preference proxy using CLIP + aesthetics

### 🖼️ Technical Image Quality Metrics  
These metrics evaluate image quality independent of the text prompt:
- **BRISQUE**: Blind/Referenceless Image Spatial Quality Evaluator  
- **NIQE**: Natural Image Quality Evaluator  
- **CLIP-IQA**: CLIP-based Image Quality Assessment

### 🛡️ T2ISafety Framework
- **Toxicity Safety**: Harmful content detection (hate speech, violence, NSFW)
- **Fairness**: Bias and stereotyping assessment  
- **Privacy Safety**: Privacy concerns (identifiable data, personal info)

## Evaluation Metrics

```
gradio-demo
├── src
│   ├── app.py             # Gradio frontend, loads .env, calls backend
│   ├── openai_service.py  # Azure OpenAI DALL-E 3 image generation logic
│   └── utils.py           # (Optional) Utility functions
├── requirements.txt       # List of dependencies
├── .env                   # Environment variables (not committed)
├── .env.example           # Example environment file (no secrets)
└── README.md              # Project documentation
```

## Installation

1. Clone the repository and install the required dependencies:

```powershell
git clone <repository-url>
cd webapp/gradio-demo
python -m pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your Azure OpenAI API key:

```powershell
copy .env.example .env
# Edit .env and set AZURE_OPENAI_API_KEY
```

## Running the Application

To run the Gradio application, execute the following command:

```powershell
python src/app.py
```

This will start a local server. Access the demo frontend in your web browser at `http://localhost:7860`.

## Usage

- Enter a prompt in the Gradio interface and click "Generate" to create an image using Azure OpenAI DALL-E 3.
- The generated image will be displayed below the prompt box.
- Example prompts:
  - "A futuristic city skyline at sunset, digital art"
  - "A cat riding a skateboard in Times Square, photorealistic"
  - "A watercolor painting of a mountain landscape in spring"

## Environment Variables

The app uses a `.env` file for configuration. Example:

```
AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.cognitiveservices.azure.com/
OPENAI_API_VERSION=2024-04-01-preview
DEPLOYMENT_NAME=dall-e-3
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
```

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.

---

**Security Note:**
- Do not commit your `.env` file with real API keys to version control. Use `.env.example` for sharing variable names only.