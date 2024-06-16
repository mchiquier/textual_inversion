import inspect
import random
import torch
import torch.utils.checkpoint as checkpoint
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from diffusers import (
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.training_utils import cast_training_params
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
from typing import Optional, Union

from ldm.modules.embedding_manager import EmbeddingManager
from ldm.modules.encoders.modules import FrozenCLIPEncoder
from magnification.textual_inversion_config import EmbeddingManagerConfig

import torch.nn as nn
import torch.nn.functional as F


class InstructInversion(nn.Module):
    def __init__(
        self, embedding_config: EmbeddingManagerConfig, conditioning_dropout_prob: float
    ) -> None:
        super().__init__()
        self.conditioning_dropout_prob = conditioning_dropout_prob

        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, safety_checker=None
        )
        self.noise_scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.text_encoder = FrozenCLIPEncoder().requires_grad_(False)
        self.vae = self.pipe.vae.requires_grad_(False)
        self.unet = self.pipe.unet.requires_grad_(False)

        self.embedding_manager = EmbeddingManager(
            **embedding_config, embedder=self.text_encoder
        ).requires_grad_(True)

    def forward(
        self,
        image: torch.Tensor,
        edited_image: torch.Tensor,
        prompt: Union[str, list[str]],
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
    ):
        edited_image = self.pipe.image_processor.preprocess(edited_image)
        latents = self.vae.encode(edited_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Get the text embedding for conditioning.
        prompt_embeds = self.text_encoder.encode(
            prompt, embedding_manager=self.embedding_manager
        )

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
        # In DDIM and DDPM it similar to q_sample implementation of the original diffusion repo
        noise_scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        if self.conditioning_dropout_prob is not None:
            random_p = torch.rand(bsz, device=latents.device, generator=generator)
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
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
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
            self.text_encoder.encode(prompt, embedding_manager=self.embedding_manager)
            if prompt_embeds is None
            else prompt_embeds
        )
        if self.do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            uncond_embeds = self.text_encoder.encode(
                uncond_tokens, embedding_manager=self.embedding_manager
            )
            prompt_embeds = torch.cat([prompt_embeds, uncond_embeds, uncond_embeds])

        # 3. Preprocess image
        image = self.pipe.image_processor.preprocess(image)

        # 4. set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.noise_scheduler.timesteps

        # 5. Prepare Image latents
        image_latents = self.prepare_image_latents(image)

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
        for t in tqdm(timesteps, total=len(timesteps)):
            # Expand the latents if we are doing classifier free guidance.
            # The latents are expanded 3 times because for pix2pix the guidance\
            # is applied for both the text and the input image.
            latent_model_input = (
                torch.cat([latents] * 3)
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
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(
                    3
                )
                noise_pred = (
                    noise_pred_uncond
                    + self.guidance_scale * (noise_pred_text - noise_pred_image)
                    + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
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

    def prepare_image_latents(self, image):

        if image.shape[1] == 4:
            image_latents = image
        else:
            image_latents = self.vae.encode(image).latent_dist.mode()

        image_latents = torch.cat([image_latents], dim=0)

        if self.do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat(
                [image_latents, image_latents, uncond_image_latents], dim=0
            )

        return image_latents

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


class InstructInversionBPTT(InstructInversion):
    def __init__(
        self, embedding_config: EmbeddingManagerConfig, conditioning_dropout_prob: float
    ) -> None:
        super().__init__(embedding_config, conditioning_dropout_prob)
        self.noise_scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def find_all_linear_names(self):
        # lora_module_names = set()
        # for name, module in self.unet.named_modules():
        #     if isinstance(module, nn.Linear):
        #         names = name.split(".")
        #         linear_name = names[-1]
        #         if len(names) == 1:
        #             linear_name = names[0]
        #         # if str(linear_name).isdigit():
        #         if str(linear_name).isdigit() or "emb" in linear_name:
        #             continue

        #         lora_module_names.add(linear_name)
        lora_module_names = ["to_k", "to_q", "to_v", "to_out.0"]
        return list(lora_module_names)

    def apply_lora(self, rank: int = 4) -> list:
        lora_module_names = self.find_all_linear_names()
        peft_config = LoraConfig(r=rank, target_modules=lora_module_names)
        self.unet.add_adapter(peft_config)
        if self.unet.dtype == torch.float16:
            cast_training_params(self.unet, dtype=torch.float32)
        lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        return lora_layers
        # self.unet = get_peft_model(self.unet, peft_config)
        # self.unet.print_trainable_parameters()

    def forward(
        self,
        image: torch.Tensor,
        edited_image: torch.Tensor,
        prompt: Union[str, list[str]],
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        eta: float = 0.0,
        grad_checkpoint: bool = True,
        truncated_backprop: bool = False,
        truncated_backprop_minmax: Union[tuple, list] = (35, 45),
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
    ):
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        edited_image = self.pipe.image_processor.preprocess(edited_image)
        edited_latents = self.vae.encode(edited_image).latent_dist.sample()
        edited_latents = edited_latents * self.vae.config.scaling_factor

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and (
            isinstance(prompt, list) or isinstance(prompt, tuple)
        ):
            batch_size = len(prompt)

        # Encode input prompt
        prompt_embeds = self.text_encoder.encode(
            prompt, embedding_manager=self.embedding_manager
        )
        if self.do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            uncond_embeds = self.text_encoder.encode(
                uncond_tokens, embedding_manager=self.embedding_manager
            )
            prompt_embeds = torch.cat([prompt_embeds, uncond_embeds, uncond_embeds])

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        image = self.pipe.image_processor.preprocess(image)
        image_latents = self.prepare_image_latents(image)

        height, width = image_latents.shape[-2:]
        height = height * self.vae.config.scaling_factor
        width = width * self.vae.config.scaling_factor

        # Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            image_latents.device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        self._num_timesteps = len(timesteps)

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t = torch.tensor([t], device=latents.device)
            # t = t.repeat(batch_size)

            # Expand the latents if we are doing classifier free guidance.
            # The latents are expanded 3 times because for pix2pix the guidance\
            # is applied for both the text and the input image.
            latent_model_input = (
                torch.cat([latents] * 3)
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

            if grad_checkpoint:
                noise_pred = checkpoint.checkpoint(
                    self.unet,
                    scaled_latent_model_input,
                    t,
                    prompt_embeds,
                    use_reentrant=False,
                ).sample
            else:
                noise_pred = self.unet(
                    scaled_latent_model_input, t, prompt_embeds
                ).sample

            if truncated_backprop:
                timestep = random.randint(
                    truncated_backprop_minmax[0],
                    truncated_backprop_minmax[1],
                )
                if i < timestep:
                    noise_pred = noise_pred.detach()

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(
                    3
                )
                noise_pred = (
                    noise_pred_uncond
                    + self.guidance_scale * (noise_pred_text - noise_pred_image)
                    + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )

            latents = self.noise_scheduler.step(
                noise_pred, t[0].long(), latents, **extra_step_kwargs
            ).prev_sample

        loss = F.mse_loss(latents.float(), edited_latents.float(), reduction="mean")

        # output_image = self.vae.decode(
        #     latents / self.vae.config.scaling_factor, return_dict=False
        # )[0]
        # output_image = self.pipe.image_processor.postprocess(
        #     output_image, output_type="pt"
        # )

        # loss = F.mse_loss(output_image, edited_image, reduction="mean")
        return loss


class InstructInversionClf(InstructInversionBPTT):
    def __init__(
        self, embedding_config: EmbeddingManagerConfig, conditioning_dropout_prob: float
    ) -> None:
        super().__init__(embedding_config, conditioning_dropout_prob)
        clf_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).requires_grad_(False)
        clf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clf = DiffCLIPModel(clf_model, clf_processor)

    def forward(
        self,
        image: torch.Tensor,
        edited_image: torch.Tensor,
        prompt: Union[str, list[str]],
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        eta: float = 0.0,
        grad_checkpoint: bool = True,
        truncated_backprop: bool = False,
        truncated_backprop_minmax: Union[tuple, list] = (35, 45),
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
    ):
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        edited_image = self.pipe.image_processor.preprocess(edited_image)
        edited_latents = self.vae.encode(edited_image).latent_dist.sample()
        edited_latents = edited_latents * self.vae.config.scaling_factor

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and (
            isinstance(prompt, list) or isinstance(prompt, tuple)
        ):
            batch_size = len(prompt)

        # Encode input prompt
        prompt_embeds = self.text_encoder.encode(
            prompt, embedding_manager=self.embedding_manager
        )
        if self.do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            uncond_embeds = self.text_encoder.encode(
                uncond_tokens, embedding_manager=self.embedding_manager
            )
            prompt_embeds = torch.cat([prompt_embeds, uncond_embeds, uncond_embeds])

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        image = self.pipe.image_processor.preprocess(image)
        image_latents = self.prepare_image_latents(image)

        height, width = image_latents.shape[-2:]
        height = height * self.vae.config.scaling_factor
        width = width * self.vae.config.scaling_factor

        # Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            image_latents.device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        self._num_timesteps = len(timesteps)

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t = torch.tensor([t], device=latents.device)

            # Expand the latents if we are doing classifier free guidance.
            # The latents are expanded 3 times because for pix2pix the guidance\
            # is applied for both the text and the input image.
            latent_model_input = (
                torch.cat([latents] * 3)
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

            if grad_checkpoint:
                noise_pred = checkpoint.checkpoint(
                    self.unet,
                    scaled_latent_model_input,
                    t,
                    prompt_embeds,
                    use_reentrant=False,
                ).sample
            else:
                noise_pred = self.unet(
                    scaled_latent_model_input, t, prompt_embeds
                ).sample

            if truncated_backprop:
                timestep = random.randint(
                    truncated_backprop_minmax[0],
                    truncated_backprop_minmax[1],
                )
                if i < timestep:
                    noise_pred = noise_pred.detach()

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(
                    3
                )
                noise_pred = (
                    noise_pred_uncond
                    + self.guidance_scale * (noise_pred_text - noise_pred_image)
                    + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )

            latents = self.noise_scheduler.step(
                noise_pred, t[0].long(), latents, **extra_step_kwargs
            ).prev_sample

        output_image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        output_image = self.pipe.image_processor.postprocess(
            output_image, output_type="pt"
        )

        logits_per_image = self.clf(output_image)
        logits_dog = logits_per_image[..., 1]
        targets = torch.zeros((batch_size,)).to(logits_dog.device)

        m = nn.Sigmoid()
        loss_fn = nn.BCELoss()

        loss = loss_fn(m(logits_dog), targets)
        return loss


class DiffCLIPModel(nn.Module):
    # Custom module for handling differential CLIP with preprocessing
    def __init__(self, clip_model: CLIPModel, processor: CLIPProcessor):
        super(DiffCLIPModel, self).__init__()
        self.clip_model = clip_model
        self.processor = processor

    def forward(
        self,
        image: torch.Tensor,
        texts: Optional[list] = ["a photo of a cat", "a photo of a dog"],
    ):
        # inputs = self.processor(images=image, return_tensors="pt", do_rescale=False).to(image.device)
        transform = transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BICUBIC
        )
        resized_image = transform(image)
        inputs = {"pixel_values": resized_image}

        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(
            image.device
        )

        inputs.update(text_inputs)

        outputs = self.clip_model(**inputs)
        return outputs.logits_per_image
