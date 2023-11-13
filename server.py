import random
from typing import Optional

from fastapi import HTTPException, Response
from modal import Image, Stub, gpu, method, web_endpoint
from pydantic import BaseModel

from constants import CIVTAI_LORAS, KEEP_WARM, MODEL

### Setup ###


def download_models():
    import requests
    from diffusers import DiffusionPipeline, LCMScheduler

    pipe = DiffusionPipeline.from_pretrained(
        MODEL, variant="fp16",  safety_checker=None)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    import os
    os.makedirs("loras", exist_ok=True)
    for id in CIVTAI_LORAS:
        print(f"Starting download for lora {CIVTAI_LORAS[id]} with id {id}")
        url = f"https://civitai.com/api/download/models/{id}"
        r = requests.get(url)
        with open(f"loras/{CIVTAI_LORAS[id]}.safetensors", "wb") as f:
            f.write(r.content)


image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "diffusers[torch]~=0.23.0",
        "ftfy",
        "torch",
        "torchvision",
        "transformers~=4.35.0",
        "triton",
        "safetensors",
        "torch>=2.0",
        "compel"
    )
    .pip_install("xformers", pre=True)
    .run_function(
        download_models
    )
)

stub = Stub("image-gen", image=image)

### Inference ###


@stub.cls(gpu=gpu.T4(count=1), keep_warm=KEEP_WARM)
class Model:
    def __enter__(self):
        import torch
        from compel import Compel
        from diffusers import DiffusionPipeline, LCMScheduler

        torch.backends.cuda.matmul.allow_tf32 = True

        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL, variant="fp16", safety_checker=None)

        self.pipe.scheduler = LCMScheduler.from_config(
            self.pipe.scheduler.config)

        self.compel = Compel(tokenizer=self.pipe.tokenizer,
                             text_encoder=self.pipe.text_encoder)

        self.pipe.to(device="cuda", dtype=torch.float16)

    @method()
    def inference(self, prompt, n_steps=4, cfg=1, negative_prompt=None, loras={}):

        conditioning = self.compel.build_conditioning_tensor(prompt)

        negative_conditioning = None
        if negative_prompt:
            negative_conditioning = self.compel.build_conditioning_tensor(
                negative_prompt)

        if loras:
            for name in loras:
                print(f"Applying lora {name} with weight {loras[name]}")
                self.pipe.load_lora_weights(
                    "loras", weight_name=f"{name}.safetensors")
                self.pipe.fuse_lora(lora_scale=float(loras[name]))

        image = self.pipe(
            prompt_embeds=conditioning,
            negative_prompt_embeds=negative_conditioning,
            num_inference_steps=n_steps,
            guidance_scale=cfg,
        ).images[0]

        import io

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()
        self.pipe.unload_lora_weights()

        return image_bytes


### Web endpoint ###

class InferenceRequest(BaseModel):
    prompt: str
    cfg: Optional[int] = 1
    n_steps: Optional[int] = 4
    negative_prompt: Optional[str] = None
    loras: dict = {}


@stub.function()
@web_endpoint(method="POST", label=f"image-gen-{random.randint(1, 10000)}")
async def predict(body: InferenceRequest):
    prompt = body.prompt

    for name in body.loras:
        if name not in CIVTAI_LORAS.values():
            raise HTTPException(
                status_code=400, detail=f"Invalid LORA: {name}")

    image_bytes = Model().inference.remote(prompt, n_steps=body.n_steps, cfg=body.cfg,
                                           negative_prompt=body.negative_prompt, loras=body.loras)

    return Response(content=image_bytes, media_type="image/png")


@stub.local_entrypoint()
def main(prompt: str):
    # trigger inference in cli
    # eg. modal run server.py --prompt "1girl"
    image_bytes = Model().inference.remote(prompt)
    output_path = "z_output.png"
    print(f"Saving result to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)
