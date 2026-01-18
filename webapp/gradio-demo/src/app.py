import gradio as gr
from openai_service import generate_image
from dotenv import load_dotenv
import os

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

def infer(prompt):
    try:
        image_url = generate_image(prompt)
        return image_url
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Azure DALL-E 3 Text-to-Image Generator")
    prompt = gr.Textbox(label="Enter your image prompt", placeholder="Describe the image you want to generate...")
    
    # Sample prompts
    gr.Markdown("**Try these sample prompts:**")
    with gr.Row():
        sample1 = gr.Button("A cat riding a skateboard.", size="sm")
        sample2 = gr.Button("A portrait of a woman with blue hair.", size="sm")
        sample3 = gr.Button("A scenic landscape of mountains and a river at sunset.", size="sm")
    with gr.Row():
        sample4 = gr.Button("A realistic photo of a dog in a park.", size="sm")
        sample5 = gr.Button("A watercolor painting of a forest.", size="sm")
    
    output = gr.Image(label="Generated Image")
    btn = gr.Button("Generate")

    # Click handlers for sample prompts
    sample1.click(lambda: "A cat riding a skateboard.", outputs=prompt)
    sample2.click(lambda: "A portrait of a woman with blue hair.", outputs=prompt)
    sample3.click(lambda: "A scenic landscape of mountains and a river at sunset.", outputs=prompt)
    sample4.click(lambda: "A realistic photo of a dog in a park.", outputs=prompt)
    sample5.click(lambda: "A watercolor painting of a forest.", outputs=prompt)
    
    btn.click(fn=infer, inputs=prompt, outputs=output)

if __name__ == "__main__":
    demo.launch()