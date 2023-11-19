
import os
import re

import gradio as gr
import torch
from compel import Compel, DiffusersTextualInversionManager
from diffusers import DiffusionPipeline, LCMScheduler

"""
    Hacked together Gradio for testing models and LoRAs locally with similar process to the modal one
    requires cuda, gradio, compel, diffusers etc.
"""

model_id = "Disty0/LCM_SoteMix"

pipe = DiffusionPipeline.from_pretrained(
    model_id, variant="fp16", safety_checker=None, use_safetensors=True)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_textual_inversion("./loras/FastNegativeV2.pt", "FastNegativeV2")
textual_inversion_manager = DiffusersTextualInversionManager(pipe)
pipe.to(device="cuda", dtype=torch.float16)

compel = Compel(tokenizer=pipe.tokenizer,
                text_encoder=pipe.text_encoder, truncate_long_prompts=False, textual_inversion_manager=textual_inversion_manager)


image_history = []


# load filenames frin loras folder
loras = []

for filename in os.listdir("./loras"):
    if filename.endswith(".safetensors"):
        loras.append(filename[:-12])
        pipe.load_lora_weights(
            "loras", weight_name=f"{filename[:-12]}.safetensors", adapter_name=filename[:-12])


def process_and_extract(prompt):
    global loras

    matches = re.findall(r'<([^:]+):(\d+(?:\.\d+)?)>', prompt)
    if not matches:
        return {}, prompt
    request_loras = {}

    for name, weight_str in matches:
        weight = float(weight_str)
        placeholder = f'<{re.escape(name)}:{re.escape(weight_str)}>'
        prompt = prompt.replace(placeholder, '')
        if name not in loras:
            print(f"unknown lora {name}")
        request_loras[name] = weight

    return request_loras, prompt


def generate_images(prompt, negative_prompt, num_inference_steps, guidance_scale, batch_size, seed):
    global image_history

    b4prompt = prompt
    request_loras, prompt = process_and_extract(prompt)
    with torch.no_grad():
        conditioning = compel.build_conditioning_tensor(prompt)
        negative_conditioning = compel.build_conditioning_tensor(negative_prompt)
        [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning])

    pipe.set_adapters(list(request_loras.keys()), list(request_loras.values()))
    # pipe.fuse_lora()

    if seed == -1:
        gen = None
    else:
        gen = torch.Generator().manual_seed(int(seed))

    images_list = []

    images = pipe(
        prompt_embeds=conditioning,
        negative_prompt_embeds=negative_conditioning,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=batch_size,
        generator=gen,
        output_type="pil"
    ).images

    image_history.extend(
        [(image, f"prompt: {b4prompt} \n seed: {str(seed)}") for image in images])
    images_list.extend([(image, "") for image in images])

    [conditioning, negative_conditioning] = [None, None]
    # pipe.unfuse_lora()
    # pipe.set_adapters(adapter_names=[], adapter_weights=[])

    return images_list, image_history


with gr.Blocks() as demo:
    with gr.Row():
        # Input components
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        "(masterpiece, best quality, highres)",
                        label="Enter Prompt")
                    neg_prompt_input = gr.Textbox(
                        "FastNegativeV2", label="Enter Negative Prompt")
                    inference_steps_slider = gr.Slider(
                        1, 20, value=8, label="Number of Inference Steps", step=1)
                    guidance_scale_slider = gr.Slider(
                        0, 5, value=2, label="Guidance Scale", step=0.1)
                    batch_size_input = gr.Slider(
                        1, 10, value=1, label="Batch Size", step=1)
                    seed_input = gr.Number(-1, label="Seed")
                    generate_button = gr.Button(
                        "Generate Images")

        with gr.Column():
            generated_images_output = gr.Gallery()
            feedback_output = gr.Gallery()

    generate_button.click(generate_images,
                          inputs=[prompt_input, neg_prompt_input, inference_steps_slider,
                                  guidance_scale_slider, batch_size_input, seed_input],
                          outputs=[generated_images_output, feedback_output])

if __name__ == "__main__":
    demo.launch()
