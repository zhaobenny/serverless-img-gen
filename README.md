# serverless-img-gen

> fast serverless ai image gen as an api

uses modal.com to generate images in ~2s from POST request (cold start not included)

## Usage
```
pip install modal
modal token new
modal deploy server.py
curl -X POST -H "Content-Type: application/json" -d '{"prompt":"1girl"}' <URL> -o output.png
```

## Credits
 - [LCM](https://latent-consistency-models.github.io/)
 - [Disty0/LCM_SoteMix](https://huggingface.co/Disty0/LCM_SoteMix)