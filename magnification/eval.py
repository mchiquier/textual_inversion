import torch
import pyrallis
from dataclasses import asdict
from torchvision import transforms
from torch.utils.data import DataLoader
from magnification.utils.viz import plot_grid
from magnification.textual_inversion_config import EvalConfig
from magnification.datasets import TextualInversionEval
from magnification.models.instruct_inversion import InstructInversionBPTT, InstructInversionClf


@pyrallis.wrap()
def main(cfg: EvalConfig):
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    placeholder_strings = cfg.diffusion.embedding_config.placeholder_strings

    transform = transforms.Compose([transforms.Resize(cfg.dataset.img_size)])
    eval_dataset = TextualInversionEval(
        cfg.dataset.eval_dir,
        placeholder_strings,
        transform=transform,
    )
    eval_data_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size)

    pipeline = InstructInversionClf(**asdict(cfg.diffusion))
    pipeline.embedding_manager.load(cfg.ckpt_path)
    pipeline.to(device)

    # eval loop
    print("evaluation")
    for i, token in enumerate(placeholder_strings):
        init_word = cfg.diffusion.embedding_config.initializer_words[i]
        for batch in eval_data_loader:
            image, prompt = batch
            prompt = tuple([x[i] for x in prompt])
            image = image.to(device)
            sample = pipeline.sample(
                image,
                prompt,
                output_type="pt",
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                image_guidance_scale=cfg.image_guidance_scale
            )
            grid = torch.cat([image, sample])
            eval_batch_save_path = cfg.output_dir / f"{cfg.output_dir.name}_{token}_{init_word}.jpg"
            plot_grid(grid, eval_batch_save_path, nrow=cfg.batch_size // 4)
            break

    for batch in eval_data_loader:
        image, prompt = batch
        image = image.to(device)
        sample = pipeline.sample(
            image,
            prompt,
            output_type="pt",
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            image_guidance_scale=cfg.image_guidance_scale
        )
        grid = torch.cat([image, sample])
        eval_batch_save_path = cfg.output_dir / f"{cfg.output_dir.name}_all.jpg"
        plot_grid(grid, eval_batch_save_path, nrow=cfg.batch_size // 4)
        break


if __name__ == "__main__":
    main()
