import torch
import pyrallis
from dataclasses import asdict
from torchvision import transforms
from torch.utils.data import DataLoader
from magnification.utils.viz import plot_grid
from magnification.textual_inversion_config import EvalConfig
from magnification.datasets import TextualInversionEval
from magnification.models.instruct_inversion import InstructInversionBPTT


@pyrallis.wrap()
def main(cfg: EvalConfig):
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize(cfg.dataset.img_size)])
    eval_dataset = TextualInversionEval(
        cfg.dataset.eval_dir,
        cfg.diffusion.embedding_config.placeholder_strings,
        transform=transform,
    )
    eval_data_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size)

    pipeline = InstructInversionBPTT(**asdict(cfg.diffusion))
    pipeline.embedding_manager.load(cfg.ckpt_path)
    pipeline.to(device)

    # eval loop
    print("evaluation")
    for batch in eval_data_loader:
        image, prompt = batch
        image = image.to(device)
        sample = pipeline.sample(
            image,
            prompt,
            output_type="pt",
            num_inference_steps=cfg.num_inference_steps,
        )
        grid = torch.cat([image, sample])
        eval_batch_save_path = cfg.output_dir / f"{cfg.run_name}.jpg"
        plot_grid(grid, eval_batch_save_path, nrow=cfg.batch_size)
        break


if __name__ == "__main__":
    main()
