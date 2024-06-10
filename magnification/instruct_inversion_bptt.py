import torch
import wandb
import random
import pyrallis
from dataclasses import asdict
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from magnification.utils.viz import plot_grid
from magnification.datasets import TextualInversionEdits, TextualInversionEval
from magnification.textual_inversion_config import InstructInversionBPTTConfig
from magnification.models.instruct_inversion import InstructInversionBPTT


@pyrallis.wrap()
def main(cfg: InstructInversionBPTTConfig):
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.Resize(cfg.dataset.img_size), transforms.RandomHorizontalFlip()]
    )
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
    data_loader = DataLoader(
        dataset, batch_size=cfg.train.total_batch_size, shuffle=True
    )
    eval_dataset = TextualInversionEval(
        cfg.dataset.eval_dir,
        cfg.diffusion.embedding_config.placeholder_strings,
        transform=transform,
    )
    eval_data_loader = DataLoader(eval_dataset, batch_size=cfg.train.total_batch_size)

    accelerator_config = ProjectConfiguration(
        project_dir=cfg.log_dir / cfg.run_name,
        automatic_checkpoint_naming=True,
        total_limit=cfg.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=cfg.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        wandb_args = {}
        if cfg.debug:
            wandb_args = {"mode": "disabled"}
        accelerator.init_trackers(
            project_name="instruct-inversion-bptt",
            config=asdict(cfg),
            init_kwargs={"wandb": wandb_args},
        )

        accelerator.project_configuration.project_dir = str(
            cfg.log_dir / wandb.run.name
        )
        accelerator.project_configuration.logging_dir = str(
            cfg.log_dir / wandb.run.name
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    set_seed(42, device_specific=True)

    pipeline = InstructInversionBPTT(**asdict(cfg.diffusion))

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    # pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    # pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    # pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    pipeline.vae.to(device, dtype=inference_dtype)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    pipeline.unet.to(device, dtype=inference_dtype)

    # Set lora layers
    if cfg.train.use_lora:
        pipeline.set_peft_unet()

    # Initialize the optimizer
    embedding_params = list(pipeline.embedding_manager.embedding_parameters())
    optimizer = torch.optim.AdamW(embedding_params, lr=cfg.train.learning_rate)

    for epoch in range(cfg.epochs):
        print(f"epoch {epoch}:")
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()

            image, image_edit, prompt = batch
            image = image.to(device).half()
            image_edit = image_edit.to(device).half()

            loss = pipeline(
                image=image,
                edited_image=image_edit,
                prompt=prompt,
                num_inference_steps=cfg.num_inference_steps,
                grad_checkpoint=cfg.train.grad_checkpoint,
            )
            print(f"batch {i} - loss: {loss.item()}")
            loss.backward(loss)
            optimizer.step()

        # save embeddings
        epoch_output_dir = cfg.log_dir / str(epoch)
        epoch_output_dir.mkdir(exist_ok=True)
        pipeline.embedding_manager.save(epoch_output_dir / f"embedding_{epoch}.pt")
        embed_mean = list(pipeline.embedding_manager.embedding_parameters())[0].mean()
        grads = [
            (name, param.grad.shape)
            for name, param in pipeline.named_parameters()
            if param.grad is not None
        ]
        print(f"learned params: {grads}")
        print(f"embed-mean: {embed_mean}")

        # visualize samples
        sample = pipeline.sample(
            image, prompt, output_type="pt", num_inference_steps=cfg.num_inference_steps
        )
        train_batch_save_path = epoch_output_dir / f"{loss.item()}_train.jpg"
        plot_grid(sample, train_batch_save_path)

        # eval loop
        print("evaluation")
        for batch in eval_data_loader:
            image, prompt = batch
            image = image.to(device).half()
            sample = pipeline.sample(
                image,
                prompt,
                output_type="pt",
                num_inference_steps=cfg.num_inference_steps,
            )
            eval_batch_save_path = epoch_output_dir / f"{loss.item()}_eval.jpg"
            plot_grid(sample, eval_batch_save_path)
            break


if __name__ == "__main__":
    main()
