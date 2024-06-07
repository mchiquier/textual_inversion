import torch
import pyrallis
from dataclasses import asdict
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from magnification.models.instruct_inversion import InstructInversion
from magnification.utils.viz import plot_grid
from magnification.datasets import TextualInversionEdits, TextualInversionEval
from magnification.textual_inversion_config import (
    TextualInversionConfig,
)


@pyrallis.wrap()
def main(cfg: TextualInversionConfig):
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
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    eval_dataset = TextualInversionEval(
        cfg.dataset.eval_dir,
        cfg.diffusion.embedding_config.placeholder_strings,
        transform=transform,
    )
    eval_data_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size)

    generator = torch.Generator(device).manual_seed(0)
    ldm = InstructInversion(**asdict(cfg.diffusion)).eval().to(device)
    optimizer = torch.optim.AdamW(ldm.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        print(f"epoch {epoch}:")
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()

            image, image_edit, prompt = batch
            image = image.to(device)
            image_edit = image_edit.to(device)

            loss = ldm(image, image_edit, prompt, generator=generator)
            print(f"batch {i} - loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        # save embeddings
        epoch_output_dir = cfg.output_dir / str(epoch)
        epoch_output_dir.mkdir(exist_ok=True)
        ldm.embedding_manager.save(epoch_output_dir / f"embedding_{epoch}.pt")
        embed_mean = list(ldm.embedding_manager.embedding_parameters())[0].mean()
        grads = [
            (name, param.grad.shape)
            for name, param in ldm.named_parameters()
            if param.grad is not None
        ]
        print(f"learned params: {grads}")
        print(f"embed-mean: {embed_mean}")

        # visualize samples
        sample = ldm.sample(
            image,
            prompt,
            output_type="pt",
            num_inference_steps=cfg.num_inference_steps,
            generator=generator,
        )
        train_batch_save_path = epoch_output_dir / f"{loss.item()}_train.jpg"
        plot_grid(sample, train_batch_save_path)

        # eval loop
        for batch in eval_data_loader:
            image, prompt = batch
            image = image.to(device)
            sample = ldm.sample(
                image,
                prompt,
                output_type="pt",
                num_inference_steps=cfg.num_inference_steps,
                generator=generator,
            )
            eval_batch_save_path = epoch_output_dir / f"{loss.item()}_eval.jpg"
            plot_grid(sample, eval_batch_save_path)
            break


if __name__ == "__main__":
    main()
