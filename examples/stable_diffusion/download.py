from diffusers import StableDiffusionPipeline  # type: ignore
import torch

StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
