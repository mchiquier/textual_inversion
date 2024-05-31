from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import numpy as np
from magnification.prompt2prompt import ptp_utils
from magnification.prompt2prompt.attention_controllers import (
    AttentionReplace,
    AttentionStore,
    EmptyControl,
)

LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


def aggregate_attention(
    attention_store: AttentionStore,
    prompts,
    res: int,
    from_where: List[str],
    is_cross: bool,
    select: int,
):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[
                    select
                ]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(
    attention_store: AttentionStore,
    res: int,
    prompts,
    tokenizer,
    from_where: List[str],
    select: int = 0,
):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(
        attention_store=attention_store,
        prompts=prompts,
        res=res,
        from_where=from_where,
        is_cross=True,
        select=select,
    )
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))


def show_self_attention_comp(
    attention_store: AttentionStore,
    res: int,
    from_where: List[str],
    max_com=10,
    select: int = 0,
):
    attention_maps = (
        aggregate_attention(attention_store, res, from_where, False, select)
        .numpy()
        .reshape((res**2, res**2))
    )
    u, s, vh = np.linalg.svd(
        attention_maps - np.mean(attention_maps, axis=1, keepdims=True)
    )
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


def run_and_display(
    model, prompts, controller, latent=None, run_baseline=False, generator=None
):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(
            prompts,
            EmptyControl(),
            latent=latent,
            run_baseline=False,
            generator=generator,
        )
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(
        model,
        prompts,
        controller,
        latent=latent,
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        low_resource=LOW_RESOURCE,
    )
    ptp_utils.view_images(images)
    return images, x_t


def main():
    device_id = 4
    device = (
        torch.device(f"cuda:{device_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4"
    ).to(device)
    tokenizer = ldm_stable.tokenizer

    g_cpu = torch.Generator().manual_seed(8888)
    prompts = [
        "A painting of a squirrel eating a burger",
        "A painting of a lion eating a burger",
    ]
    controller = AttentionStore()
    image, x_t = run_and_display(
        model=ldm_stable,
        prompts=prompts,
        controller=controller,
        latent=None,
        run_baseline=False,
        generator=g_cpu,
    )
    controller = AttentionReplace(
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        tokenizer=tokenizer,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        device=device
    )
    _ = run_and_display(
        model=ldm_stable,
        prompts=prompts,
        controller=controller,
        latent=x_t,
        run_baseline=False,
    )


if __name__ == "__main__":
    main()
