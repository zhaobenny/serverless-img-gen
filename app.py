import re
from typing import Optional

from fastapi import Depends, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from modal import Image, Secret, Stub, gpu, method, web_endpoint
from pydantic import BaseModel

from config import AUTH_TOKEN, EXTRA_URL, KEEP_WARM, MODEL

### Setup ###


def download_models():
    import requests
    from diffusers import DiffusionPipeline, LCMScheduler

    pipe = DiffusionPipeline.from_pretrained(
        MODEL, variant="fp16",  safety_checker=None)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    import os
    os.makedirs("loras", exist_ok=True)

    if not os.path.exists("loras/FastNegativeV2.pt"):
        # hardcode a single negative embedding
        r = requests.get(
            "https://civitai.com/api/download/models/94057?type=Model&format=PickleTensor")
        with open(f"loras/FastNegativeV2.pt", "wb") as f:
            f.write(r.content)

    if not os.path.exists("loras/Misato.safetensors"):
        # hardcode a single LoRA for demo purposes
        r = requests.get(
            "https://civitai.com/api/download/models/181315?type=Model&format=PickleTensor")
        with open(f"loras/Misato.safetensors", "wb") as f:
            f.write(r.content)

    print("###############")
    print("Imported the following LoRAs:")
    for file in os.listdir("loras"):
        if file.endswith(".safetensors"):
            print(f"- {file[:-12]}")
    print("###############")


image = (
    Image.debian_slim(python_version="3.10")
    .copy_local_dir(local_path="loras/", remote_path="/root/loras")
    .pip_install(
        "python-dotenv",
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

stub = Stub("sd-image-gen", image=image)

### Inference ###


@stub.cls(gpu=gpu.T4(count=1), keep_warm=KEEP_WARM)
class Model:
    def __enter__(self):
        import torch
        from compel import Compel, DiffusersTextualInversionManager
        from diffusers import DiffusionPipeline, LCMScheduler

        torch.backends.cuda.matmul.allow_tf32 = True

        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL, variant="fp16", safety_checker=None)

        self.pipe.scheduler = LCMScheduler.from_config(
            self.pipe.scheduler.config)

        self.pipe.load_textual_inversion(
            "./loras/FastNegativeV2.pt", "FastNegativeV2")

        textual_inversion_manager = DiffusersTextualInversionManager(self.pipe)

        self.compel = Compel(tokenizer=self.pipe.tokenizer,
                             text_encoder=self.pipe.text_encoder,
                             textual_inversion_manager=textual_inversion_manager,
                             truncate_long_prompts=False)

        self.pipe.to(device="cuda", dtype=torch.float16)

    @method()
    def inference(self, prompt, n_steps=6, cfg=2, negative_prompt="", loras={}):

        conditioning = self.compel.build_conditioning_tensor(prompt)

        negative_conditioning = self.compel.build_conditioning_tensor(
            negative_prompt)

        [conditioning, negative_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning])

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
        self.pipe.unfuse_lora()

        return image_bytes


### Web endpoint ###

class InferenceRequest(BaseModel):
    prompt: str
    cfg: Optional[int] = 2
    n_steps: Optional[int] = 6
    negative_prompt: Optional[str] = None


auth_scheme = HTTPBearer()


@stub.function(secret=Secret.from_dict(AUTH_TOKEN))
@web_endpoint(method="POST", label=f"{EXTRA_URL}-imggen")
async def predict(body: InferenceRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    import os
    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=401,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    loras, prompt = await process_and_extract(body.prompt)

    image_bytes = Model().inference.remote(prompt, n_steps=body.n_steps, cfg=body.cfg,
                                           negative_prompt=body.negative_prompt, loras=loras)

    return Response(content=image_bytes, media_type="image/png")


async def process_and_extract(prompt):
    matches = re.findall(r'<([^:]+):(\d+(?:\.\d+)?)>', prompt)
    if not matches:
        return {}, prompt
    request_loras = {}
    loras_names = await get_loras_names()

    for name, weight_str in matches:
        weight = float(weight_str)
        placeholder = f'<{re.escape(name)}:{re.escape(weight_str)}>'
        prompt = prompt.replace(placeholder, '')
        if name not in loras_names:
            raise HTTPException(
                status_code=400, detail=f"Invalid LORA name {name}")
        request_loras[name] = weight

    return request_loras, prompt


async def get_loras_names():
    import os
    loras_names = []
    for file in os.listdir("loras"):
        if file.endswith(".safetensors"):
            loras_names.append(file[:-12])
    return set(loras_names)


@stub.local_entrypoint()
def main(prompt: str, steps: int = 6, output_path: str = "zlocaloutput.png"):
    # trigger inference in cli
    # eg. modal run app.py --prompt "1girl" --steps 8
    image_bytes = Model().inference.remote(
        prompt, n_steps=steps)
    print(f"Saving result to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)