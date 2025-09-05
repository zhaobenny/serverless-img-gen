import os
import re

import modal
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from modal import App, Image, Secret, asgi_app, enter, method
from pydantic import BaseModel

from config import AUTH_TOKEN, EXTRA_URL, KEEP_WARM, MODEL, NO_DEMO

### Modal setup ###


def download_models():
    import requests
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        MODEL,
        variant="fp16",
        feature_extractor=None,
        safety_checker=None,
        requires_safety_checker=False
    )

    import os
    os.makedirs("loras", exist_ok=True)

    if not NO_DEMO and not os.path.exists("loras/FastNegativeV2.pt"):
        # hardcode a negative embedding for demo purposes
        r = requests.get(
            "https://civitai.com/api/download/models/94057?type=Model&format=PickleTensor")
        with open(f"loras/FastNegativeV2.pt", "wb") as f:
            f.write(r.content)

    if not NO_DEMO and not os.path.exists("loras/Misato.safetensors"):
        # hardcode a single LoRA for demo purposes
        r = requests.get(
            "https://civitai.com/api/download/models/181315?type=Model&format=PickleTensor")
        with open(f"loras/Misato.safetensors", "wb") as f:
            f.write(r.content)

    print("\n###############")
    print("\nImported the following LoRAs:")
    for file in os.listdir("loras"):
        if file.endswith(".safetensors"):
            print(f"- {file[:-12]}")

    print("\nImported the following embeddings:")
    for file in os.listdir("loras"):
        if file.endswith(".pt"):
            print(f"- {file[:-3]}")

    print("\nAttempting test load of imported weights:\n")
    for file in os.listdir("loras"):
        if file.endswith(".safetensors"):
            print(f"Loading LoRA {file[:-12]}")
            pipe.load_lora_weights(
                "loras", weight_name=f"{file[:-12]}.safetensors", adapter_name=file[:-12])
        if file.endswith(".pt"):
            print(f"Loading embedding {file[:-3]}")
            pipe.load_textual_inversion(f"./loras/{file}", file[:-3])
    print("\n###############")


image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "python-dotenv",
        "ftfy",
        "requests"
    )
    .pip_install(
        "accelerate",
        "diffusers[torch]~=0.24.0",
        "torchvision",
        "transformers~=4.36.0",
        "triton",
        "safetensors",
        "torch>=2.0",
        "compel~=2.0.0",
        "peft~=0.7.0",
        "xformers",
    ).add_local_dir(
        local_path="loras/", remote_path="/root/loras", copy=True
    )
    .add_local_file(local_path="app.py", remote_path="/root/app.py")
    .add_local_file(local_path="config.py", remote_path="/root/config.py")
    .run_function(
        download_models
    )
)

app = App("sd-image-gen", image=image)

### Inference ###


@app.cls(gpu="T4", min_containers=KEEP_WARM)
class Model:

    @enter()
    def startup(self):
        import os

        import torch
        from compel import Compel, DiffusersTextualInversionManager
        from diffusers import DiffusionPipeline

        torch.backends.cuda.matmul.allow_tf32 = True

        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
        )

        textual_inversion_manager = DiffusersTextualInversionManager(self.pipe)

        self.compel = Compel(tokenizer=self.pipe.tokenizer,
                             text_encoder=self.pipe.text_encoder,
                             textual_inversion_manager=textual_inversion_manager,
                             truncate_long_prompts=False)

        self.pipe.to(device="cuda", dtype=torch.float16)

        for file in os.listdir("loras"):
            if file.endswith(".safetensors"):
                self.pipe.load_lora_weights("loras", weight_name=f"{file[:-12]}.safetensors", adapter_name=file[:-12])
            if file.endswith(".pt"):
                self.pipe.load_textual_inversion(f"./loras/{file}", file[:-3])

    @method()
    def inference(self, prompt, n_steps=7, cfg=2, negative_prompt="", loras={}, height=512, width=512):
        import torch
        with torch.inference_mode():
            conditioning = self.compel.build_conditioning_tensor(prompt)

            negative_conditioning = self.compel.build_conditioning_tensor(negative_prompt)

            [conditioning, negative_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
                [conditioning, negative_conditioning])

            if loras:
                self.pipe.set_adapters(list(loras.keys()), list(loras.values()))
                self.pipe.fuse_lora()

            image = self.pipe(
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                num_inference_steps=n_steps,
                guidance_scale=cfg,
                height=height,
                width=width
            ).images[0]

            import io
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                try:
                    return img_bytes
                finally:
                    self._cleanup(loras)

    def _cleanup(self, loras):
        if not loras:
            return
        self.pipe.unfuse_lora()
        self.pipe.set_adapters(adapter_names=[], adapter_weights=[])


### Web endpoint ###

class InferenceRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    cfg: int = 2
    n_steps: int = 7
    height: int = 512
    width: int = 512


class LorasResponse(BaseModel):
    loras: list[str] = []


class EmbeddingsResponse(BaseModel):
    embeddings: list[str] = []


auth_scheme = HTTPBearer()
web_app = FastAPI()
loras_names = []
embeddings_names = []

if not (modal.is_local()):
    for file in os.listdir("loras"):
        if file.endswith(".safetensors"):
            loras_names.append(file[:-12])
        if file.endswith(".pt"):
            embeddings_names.append(file[:-3])
    loras_names = set(loras_names)


@web_app.post("/", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
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
                                           negative_prompt=body.negative_prompt, loras=loras, height=body.height, width=body.width)

    return Response(content=image_bytes, media_type="image/png")


@web_app.get(path="/loras", response_model=LorasResponse)
async def get_available_loras():
    return LorasResponse(loras=list(loras_names))


@web_app.get(path="/embeddings", response_model=EmbeddingsResponse)
async def get_available_embeddings():
    return EmbeddingsResponse(embeddings=embeddings_names)


async def process_and_extract(prompt):
    matches = re.findall(r'<([^:]+):(\d+(?:\.\d+)?)>', prompt)
    if not matches:
        return {}, prompt
    request_loras = {}

    for name, weight_str in matches:
        weight = float(weight_str)
        placeholder = f'<{re.escape(name)}:{re.escape(weight_str)}>'
        prompt = prompt.replace(placeholder, '')
        if name not in loras_names:
            raise HTTPException(
                status_code=400, detail=f"Invalid LORA name {name}")
        request_loras[name] = weight

    return request_loras, prompt


@app.function(secrets=[Secret.from_dict(AUTH_TOKEN)])
@asgi_app(label=f"{EXTRA_URL}-imggen")
def fastapi_app():
    return web_app


@app.local_entrypoint()
def main(prompt: str, steps: int = 7, output_path: str = "zlocaloutput.png"):
    # trigger inference in cli
    # eg. modal run app.py --prompt "(masterpiece, best quality, highres), 1girl" --steps 8
    image_bytes = Model().inference.remote(
        prompt, n_steps=steps)
    print(f"Saving result to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)
