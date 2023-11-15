# serverless-img-gen

> fast serverless ai image gen as an api

uses modal.com to generate images in ~2s from POST request (cold start not included)

## Usage
```
git clone https://github.com/zhaobenny/serverless-img-gen.git && cd serverless-img-gen
pip install modal python-dotenv
modal token new
modal deploy app.py
curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer a-good-auth-token" -d '{"prompt":"<Misato:0.4>, (masterpiece, best quality, highres), selfie, misato katsuragi, long hair, (brown eyes), blue hair, (purple hair), solo, white shirt", "negative_prompt":"FastNegativeV2"}' <URL> -o output.png
```

## Prompting

Prompt uses Compel syntax, see their [docs](https://github.com/damian0815/compel/blob/main/doc/syntax.md)

LoRA syntax is `<Misato:0.4>` in the prompt where `Misato` is the filename of the LoRA in `loras` folder, and an suitable weight after the colon.

[FastNegativeV2](https://civitai.com/models/71961/fast-negative-embedding-fastnegativev2) embedding is available for negative prompts by default.

The [Misato LyCORIS](https://civitai.com/models/161112/misato-katsuragi-neon-genesis-evangelion) is also included by default for demo purposes.

## Credits
 - [LCM](https://latent-consistency-models.github.io/)
 - [Disty0/LCM_SoteMix](https://huggingface.co/Disty0/LCM_SoteMix)