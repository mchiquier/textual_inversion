import torch
import inspect
import pyrallis
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from dataclasses import asdict
from torchvision import transforms
from torchvision.utils import make_grid
from typing import Optional, Union
from torch.utils.data import DataLoader, ConcatDataset
from magnification.datasets import TextualInversionEdits
from ldm.modules.embedding_manager import EmbeddingManager
from ldm.modules.encoders.modules import FrozenCLIPEncoder
from magnification.textual_inversion_config import (
    EmbeddingManagerConfig,
    TextualInversionConfig,
)
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
)
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class LDM(nn.Module):
    def __init__(
        self, embedding_config: EmbeddingManagerConfig, conditioning_dropout_prob: float
    ) -> None:
        super().__init__()
        self.conditioning_dropout_prob = conditioning_dropout_prob
        self.generator = torch.Generator().manual_seed(42)

        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, safety_checker=None
        )
        self.noise_scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.noise_scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        # self.text_encoder = self.instruct_p2p.text_encoder
        self.text_encoder = FrozenCLIPEncoder()
        self.tokenizer = self.pipe.tokenizer
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        self.embedding_manager = EmbeddingManager(
            **embedding_config, embedder=self.text_encoder
        )
        self.embedding_params = list(self.embedding_manager.embedding_parameters())

    def forward(
        self,
        image: torch.Tensor,
        edited_image: torch.Tensor,
        prompt: Union[str, list[str]] = None,
    ):
        edited_image = self.pipe.image_processor.preprocess(edited_image)
        latents = self.vae.encode(edited_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Get the text embedding for conditioning.
        prompt_embeds = self.text_encoder(prompt)

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        image = self.pipe.image_processor.preprocess(image)
        original_image_embeds = self.vae.encode(image).latent_dist.mode()

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        if self.conditioning_dropout_prob is not None:
            random_p = torch.rand(bsz, device=latents.device, generator=self.generator)
            # only use image masking for classifier-free guidance

            # Sample masks for the edit prompts.
            # prompt_mask = random_p < 2 * self.conditioning_dropout_prob
            # prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            # Final text conditioning.
            # null_conditioning = self.text_encoder(tokenize_captions([""]))[0]
            # encoder_hidden_states = torch.where(
            #     prompt_mask, null_conditioning, encoder_hidden_states
            # )

            # Sample masks for the original images.
            image_mask_dtype = original_image_embeds.dtype
            image_mask = 1 - (
                (random_p >= self.conditioning_dropout_prob).to(image_mask_dtype)
                * (random_p < 3 * self.conditioning_dropout_prob).to(image_mask_dtype)
            )
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            # Final image conditioning.
            original_image_embeds = image_mask * original_image_embeds

        # Concatenate the `original_image_embeds` with the `noisy_latents`.
        concatenated_noisy_latents = torch.cat(
            [noisy_latents, original_image_embeds], dim=1
        )

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        # Predict the noise residual and compute loss
        model_pred = self.unet(
            concatenated_noisy_latents,
            timesteps,
            prompt_embeds,
            return_dict=False,
        )[0]

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        prompt: Union[str, list[str]] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        eta: float = 0.0,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        latents: Optional[torch.FloatTensor] = None,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
    ):
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and (
            isinstance(prompt, list) or isinstance(prompt, tuple)
        ):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 2. Encode input prompt
        prompt_embeds = (
            self.text_encoder(prompt) if prompt_embeds is None else prompt_embeds
        )

        # 3. Preprocess image
        image = self.pipe.image_processor.preprocess(image)

        # 4. set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.noise_scheduler.timesteps

        # 5. Prepare Image latents
        image_latents = self.vae.encode(image).latent_dist.mode()

        height, width = image_latents.shape[-2:]
        height = height * self.vae.config.scaling_factor
        width = width * self.vae.config.scaling_factor

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            image_latents.device,
            generator,
            latents,
        )

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        self._num_timesteps = len(timesteps)
        for _ in tqdm(range(num_inference_steps)):
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = (
                    torch.cat([latents] * 3, dim=1)
                    if self.do_classifier_free_guidance
                    else latents
                )

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.noise_scheduler.scale_model_input(
                    latent_model_input, t
                )
                scaled_latent_model_input = torch.cat(
                    [scaled_latent_model_input, image_latents], dim=1
                )

                # predict the noise residual
                noise_pred = self.unet(
                    scaled_latent_model_input,
                    t,
                    prompt_embeds,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = (
                        noise_pred.chunk(3)
                    )
                    noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.image_guidance_scale
                        * (noise_pred_image - noise_pred_uncond)
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]

        image = self.pipe.image_processor.postprocess(image, output_type=output_type)
        return image

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height // self.vae.config.scaling_factor),
            int(width // self.vae.config.scaling_factor),
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = torch.randn(
                shape, generator=generator, dtype=dtype, device=device
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def do_classifier_free_guidance(self):
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        return self.guidance_scale > 1.0 and self.image_guidance_scale >= 1.0


@pyrallis.wrap()
def main(cfg: TextualInversionConfig):
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")

    transform = None
    dataset = ConcatDataset(
        [
            TextualInversionEdits(
                cfg.dataset.image_dir,
                cfg.dataset.image_edits_dir,
                cfg.diffusion.embedding_config.placeholder_strings,
                transform=transform,
            )
        ]
        * cfg.dataset.repeats
    )
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    ldm = LDM(**asdict(cfg.diffusion)).to(device)
    optimizer = torch.optim.AdamW(ldm.embedding_params, lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        print(f"epoch {epoch}:")
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()

            image, image_edit, prompt = batch
            image = image.to(device)
            image_edit = image_edit.to(device)

            loss = ldm(image, image_edit, prompt)
            print(f"batch {i} - loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        # save embeddings
        epoch_output_dir = cfg.output_dir / str(epoch)
        epoch_output_dir.mkdir(exist_ok=True)
        ldm.embedding_manager.save(epoch_output_dir / f"embedding_{epoch}.pt")

        # visualize samples
        sample = ldm.sample(
            image, prompt, guidance_scale=1, num_inference_steps=2, output_type="pt"
        )
        sample = torch.clamp(sample.detach().cpu(), -1., 1)
        grid = make_grid(sample, nrow=4)
        grid = (grid + 1.0) / 2.0
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid * 255).astype(np.uint8)
        im = Image.fromarray(grid)
        filename = f"{loss.item()}_train.jpg"
        im.save(epoch_output_dir / filename)


if __name__ == "__main__":
    main()
