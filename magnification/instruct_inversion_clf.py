import torch
import wandb
import pyrallis
from dataclasses import asdict
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from magnification.utils.viz import plot_grid, plot_logits_and_predictions
from magnification.datasets import TextualInversionAFHQ, TextualInversionEval
from magnification.textual_inversion_config import InstructInversionBPTTConfig
from magnification.models.instruct_inversion import InstructInversionClf


@pyrallis.wrap()
def main(cfg: InstructInversionBPTTConfig):

    transform = transforms.Compose(
        [transforms.Resize(cfg.dataset.img_size), transforms.RandomHorizontalFlip()]
    )
    dataset = ConcatDataset(
        [
            TextualInversionAFHQ(
                root_dir=cfg.dataset.image_dir,
                placeholder_str=cfg.diffusion.embedding_config.placeholder_strings,
                transform=transform,
                split="train",
            )
        ]
        * cfg.dataset.repeats
    )

    # Option to take a subset of the training set (useful for debug)
    if cfg.dataset.subset is not None:
        subset_indices = list(range(cfg.dataset.subset))
        dataset = Subset(dataset, subset_indices)

    data_loader = DataLoader(
        dataset, batch_size=cfg.train.total_batch_size, shuffle=True
    )
    eval_dataset = TextualInversionEval(
        cfg.dataset.eval_dir,
        cfg.diffusion.embedding_config.placeholder_strings,
        transform=transforms.Resize(cfg.dataset.img_size),
    )
    eval_data_loader = DataLoader(eval_dataset, batch_size=cfg.train.total_batch_size)

    accelerator_kwargs = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=cfg.mixed_precision,
        kwargs_handlers=accelerator_kwargs,
    )
    # device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    device = accelerator.device

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

    pipeline = InstructInversionClf(**asdict(cfg.diffusion))

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.to(device, dtype=inference_dtype)

    params_to_optimize = [
        {
            "params": pipeline.embedding_manager.embedding_parameters(),
            "lr": cfg.train.lr_text_embed,
        }
    ]
    # Set lora layers
    if cfg.train.use_lora:
        lora_layers = pipeline.apply_lora()
        params_to_optimize.append({"params": lora_layers, "lr": cfg.train.lr_lora})
        pipeline.unet.train()

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(params_to_optimize)

    pipeline, optimizer, data_loader, eval_data_loader = accelerator.prepare(
        pipeline, optimizer, data_loader, eval_data_loader
    )

    for epoch in range(cfg.epochs):
        accelerator.print(f"epoch {epoch}:")

        train_loss = 0.0
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()

            image, prompt = batch
            curr_batch_size = image.shape[0]
            # image = image.to(device)
            # image_edit = image_edit.to(device)

            loss, _, _ = pipeline(
                image=image,
                edited_image=None,
                prompt=prompt,
                num_inference_steps=cfg.num_inference_steps,
                grad_checkpoint=cfg.train.grad_checkpoint,
                truncated_backprop=cfg.train.truncated_backprop,
                truncated_backprop_minmax=cfg.train.truncated_backprop_minmax,
            )
            accelerator.print(f"batch {i} - loss: {loss.item()}")
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()

            train_loss += loss.item() * curr_batch_size

        train_loss /= len(dataset)

        if accelerator.is_main_process:
            # save embeddings
            epoch_output_dir = cfg.log_dir / str(epoch)
            epoch_output_dir.mkdir(exist_ok=True)
            unwrapped_pipeline = accelerator.unwrap_model(pipeline)
            # pipeline.embedding_manager.save(epoch_output_dir / f"embedding_{epoch}.pt")
            save_dict = unwrapped_pipeline.embedding_manager.get_save_dict()
            accelerator.save(save_dict, epoch_output_dir / f"embedding_{epoch}.pt")

            embed_mean = list(
                unwrapped_pipeline.embedding_manager.embedding_parameters()
            )[0].mean()
            grads = [
                (name, param.grad.shape)
                for name, param in unwrapped_pipeline.named_parameters()
                if param.grad is not None
            ]
            accelerator.print(f"number of learned params: {len(grads)}")
            accelerator.print(f"embed-mean: {embed_mean}")

            logs = {
                "train/loss": train_loss,
                "train/embed-mean": embed_mean,
                "train/number-learned-params": len(grads),
            }

            # visualize samples
            # sample = pipeline.sample(
            #     image, prompt, output_type="pt", num_inference_steps=cfg.num_inference_steps
            # )
            # train_batch_save_path = epoch_output_dir / f"{loss.item()}_train.jpg"
            # train_grid = plot_grid(sample, train_batch_save_path)
            # logs.update({"train/grid": wandb.Image(train_grid)})

            # plot logits and pred
            # logits_plot_path = epoch_output_dir / f"{loss.item()}_logits.jpg"
            # fig = plot_logits_and_predictions(logits, probs, logits_plot_path)
            # logs.update({"logits": fig})

            # eval loop
            accelerator.print("evaluation")
            val_loss = 0.0
            for i, batch in enumerate(eval_data_loader):
                image, prompt = batch
                curr_batch_size = image.shape[0]
                with torch.no_grad():
                    val_batch_loss, _, _ = pipeline(
                        image=image,
                        edited_image=None,
                        prompt=prompt,
                        num_inference_steps=cfg.num_inference_steps,
                        grad_checkpoint=cfg.train.grad_checkpoint,
                        truncated_backprop=cfg.train.truncated_backprop,
                        truncated_backprop_minmax=cfg.train.truncated_backprop_minmax,
                    )

                    val_loss += val_batch_loss.item() * curr_batch_size
                # image = image.to(device)
                if i == 0:
                    sample = unwrapped_pipeline.sample(
                        image,
                        prompt,
                        output_type="pt",
                        num_inference_steps=cfg.num_inference_steps,
                    )
                    concat_samples = torch.cat([image, sample])
                    eval_batch_save_path = epoch_output_dir / f"{loss.item()}_eval.jpg"
                    eval_grid = plot_grid(concat_samples, eval_batch_save_path, nrow=8)
                    logs.update({"val/grid": wandb.Image(eval_grid)})
            
            val_loss /= len(eval_dataset)
            logs.update({"val/loss": val_loss})
            accelerator.log(logs)

    accelerator.end_training()


if __name__ == "__main__":
    main()
