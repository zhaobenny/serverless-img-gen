
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
reloading_loras = False


def generate_images(prompt, negative_prompt, num_inference_steps, guidance_scale, batch_size, seed):
    global image_history
    global reloading_loras

    if reloading_loras:
        return [], image_history

    conditioning = compel.build_conditioning_tensor(prompt)
    negative_conditioning = compel.build_conditioning_tensor(negative_prompt)
    [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length(
        [conditioning, negative_conditioning])

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

    image_history.extend([(image, "") for image in images])
    images_list.extend([(image, "") for image in images])

    return images_list, image_history


def load_unload_loras(loras_file, scale):
    global reloading_loras
    reloading_loras = True
    print("reloading loras")
    pipe.unfuse_lora()
    if loras_file is not None:
        pipe.load_lora_weights(loras_file, weight_name="loras.safetensors")
        pipe.fuse_lora(lora_scale=float(scale))
    reloading_loras = False
    print("done reloading weights")


with gr.Blocks() as demo:
    # Title Markdown
    gr.Markdown(
        """
        # testing
        there's a memory leak somewhere.. restart when slow
        """)

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
            with gr.Row():
                with gr.Column():
                    lora_file_input = gr.File(label="Lora File")
                    lora_scale_input = gr.Slider(
                        0, 2, label="Lora Scale", step=0.1)
                    reload_lora_button = gr.Button("Reload Lora")

        with gr.Column():
            generated_images_output = gr.Gallery()
            feedback_output = gr.Gallery()

    reload_lora_button.click(load_unload_loras,
                             inputs=[lora_file_input, lora_scale_input],
                             outputs=[])
    generate_button.click(generate_images,
                          inputs=[prompt_input, neg_prompt_input, inference_steps_slider,
                                  guidance_scale_slider, batch_size_input, seed_input],
                          outputs=[generated_images_output, feedback_output])

if __name__ == "__main__":
    demo.launch()
