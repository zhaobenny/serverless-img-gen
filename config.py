import os

from dotenv import load_dotenv

load_dotenv()

### LoRAs / embeddings ###
# Add diffusers compatible LoRAs / embeddings to the "loras" folder

# Generate your own token
AUTH_TOKEN = {"AUTH_TOKEN": os.getenv("AUTH_TOKEN") or "a-good-auth-token"}

# Put your own url friendly string here
EXTRA_URL = os.getenv("EXTRA_URL") or "great-potato-microwave"

# Keep GPU alive for int n seconds after request or disable with None
KEEP_WARM = None

# Diffusion model id (from huggingface.co/models)
MODEL = "Disty0/LCM_SoteMix"

# Skips downloading demo LoRA/embedding (for compatibility with non SD 1.5 models)
NO_DEMO = False
