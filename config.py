import os

from dotenv import load_dotenv

load_dotenv()

# add diffusers-compatible LoRAs to the "loras" folder

# generate your own token
AUTH_TOKEN = {"AUTH_TOKEN": os.getenv("AUTH_TOKEN") or "a-good-auth-token"}

# put your own url friendly string here
EXTRA_URL = os.getenv("EXTRA_URL") or "great-potato-microwave"

# keep gpu alive for int n seconds after request or disable with None
KEEP_WARM = None

# id of model converted to LCM
MODEL = "Disty0/LCM_SoteMix"
