import torch
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset

from ldm.util import instantiate_from_config
from ldm.data.personalized import  PersonalizedBase
from ldm.models.diffusion.ddpm_edit import LatentDiffusion

'''
global vars, consider making a script to set these
'''
ckpt = 'models/ldm/stable-diffusion-v1/instruct-pix2pix-00-22000.ckpt'
yml_paths = ['configs/latent-diffusion/instructpix2pix.yaml', ]
data_root = 'vases'
device=0



'''
helper functions
'''
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model

def set_lr(model, config):
    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    accumulate_grad_batches = 1
    ngpu = 1

    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    print(
        "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))


'''
initialize model
'''
# load omega conf
config_paths = [OmegaConf.load(cfg) for cfg in yml_paths]
dummy = OmegaConf.from_dotlist([])
model_config = OmegaConf.merge(*config_paths, dummy)
train_config = model_config.pop("lightning", OmegaConf.create())
# trainer_config = train_parameters.get("trainer", OmegaConf.create())
model: LatentDiffusion = load_model_from_config(model_config, ckpt)
model = model.eval().to(device)

'''
create dataloader
'''
model_config.data.params.train.params.data_root = data_root
model_config.data.params.validation.params.data_root = data_root

# annoying that we have to use lightning here
# (can do another way but will take more code)
# its ok, we just grab the dataloader from this
data = instantiate_from_config(model_config.data)
data.prepare_data()
data.setup()
train_loader = data._train_dataloader()

set_lr(model, model_config)
embedding_params = list(model.embedding_manager.embedding_parameters())#[self.parametersdiff]
optimizer = torch.optim.AdamW(embedding_params, lr=model.learning_rate)

for raw_batch in train_loader:
    # NOTE: way it is currently coded the k='edited' parameter does not do anything..

    optimizer.zero_grad()

    raw_batch['edited'] = raw_batch['edited'].to(device)
    raw_batch['image'] = raw_batch['image'].to(device)
    batch = model.get_input(raw_batch, 'edited')

    # calls p_losses
    loss, loss_dict = model(batch[0], batch[1])

    optimizer.step()

    if model.use_scheduler:
        lr = model.optimizers().param_groups[0]['lr']
        model.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

    print(loss)

    # # NOTE: probably does not matter as model not updated
    # if model.use_ema:
    #     model.model_ema(model)