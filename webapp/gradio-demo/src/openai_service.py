from dotenv import load_dotenv
import os
# Load environment variables from .env file
base_dir = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(base_dir, '.env')
load_dotenv(env_path)

from openai import AzureOpenAI
import json

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://hshuj-mkj0lbym-swedencentral.cognitiveservices.azure.com/")
api_version = os.getenv("OPENAI_API_VERSION", "2024-04-01-preview")
deployment = os.getenv("DEPLOYMENT_NAME", "dall-e-3")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

def generate_image(prompt: str) -> str:
    """
    Generate an image URL from a text prompt using Azure OpenAI DALL-E 3.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    result = client.images.generate(
        model=deployment,
        prompt=prompt,
        n=1,
        style="vivid",
        quality="standard",
    )
    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    return image_url

# Remove any top-level calls to client.images.generate or print