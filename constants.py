
MODEL = "Disty0/LCM_SoteMix"  # LCM diffusers models only for now

# find civtai id  by copying the url from the download button, NOT from the model page url itself
# civtai id : any name
CIVTAI_LORAS = {"204798": "Frieren"}
# negative loras not supported

KEEP_WARM = None # put int n to keep gpu function warm for n seconds after inference (costs more)
