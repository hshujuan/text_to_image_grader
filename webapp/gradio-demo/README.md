# Azure DALL-E 3 Text-to-Image Gradio App

This project demonstrates a Gradio application for generating images using Azure OpenAI DALL-E 3. Enter a prompt and get a generated image in seconds!

## Project Structure

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