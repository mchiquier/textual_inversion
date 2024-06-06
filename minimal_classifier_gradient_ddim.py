import torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import time
from ldm.util import instantiate_from_config
from ldm.data.personalized import  PersonalizedBase
from ldm.models.diffusion.ddpm_edit import LatentDiffusion
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
import pdb
import torchvision
import numpy as np 
from PIL import Image
import os
import matplotlib.pyplot as plt
from clip import load as clip_load
import pdb
import time

hparams = {'image_size': 224, 'device': 'cpu', 'model_size': "ViT-B/32"}
device = torch.device(hparams['device'])
model, preprocess = clip_load(hparams['model_size'], device=device, jit=False)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder = folder_path
        self.transform = transform
        self.image_paths = os.listdir(folder_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder, self.image_paths[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image
    
dataset = CustomDataset(folder_path='/proj/vondrick4/datasets/data/afhq/train/cat', transform=transform)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


'''

global vars, consider making a script to set these
'''
ckpt = 'models/ldm/stable-diffusion-v1/instruct-pix2pix-00-22000.ckpt'
yml_paths = ['configs/latent-diffusion/instructpix2pix.yaml', ]
data_root = '../../VisualMacros/Photoshop/whiskers/train-a'
edit_root = '../../VisualMacros/Photoshop/whiskers/train-b'
data_eval_root = '../../VisualMacros/Photoshop/whiskers/eval-a'
output_path = "results/whiskers_bptt_700grad" 
if not os.path.isdir(output_path):
    os.mkdir(output_path)
device=4



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
model_config.data.params.validation.params.data_root = data_eval_root

model_config.data.params.train.params.edit_root = edit_root
model_config.data.params.validation.params.edit_root = edit_root

# annoying that we have to use lightning here
# (can do another way but will take more code)
# its ok, we just grab the dataloader from this
data = instantiate_from_config(model_config.data)
data.prepare_data()
data.setup()
train_loader = data._train_dataloader()
val_loader = data._val_dataloader()
raw_eval_batch = next(iter(val_loader))
raw_eval_batch['edited'] = raw_eval_batch['edited'].to(device)
raw_eval_batch['image'] = raw_eval_batch['image'].to(device)
raw_eval_batch['caption'] = ['*'] * raw_eval_batch['image'].shape[0]

set_lr(model, model_config)
embedding_params = list(model.embedding_manager.embedding_parameters())#[self.parametersdiff]
num=100
optimizer = torch.optim.AdamW(embedding_params, lr=model.learning_rate)
raw_batch = next(iter(train_loader))
noise=None
raw_batch['edited'] = raw_batch['edited'].to(device)
raw_batch['image'] = raw_batch['image'].to(device)
raw_batch['caption'] = ['*']*raw_batch['image'].shape[0]
thestring="bptt"
batch = model.get_input(raw_batch, 'edited')
if not os.path.isdir(output_path +"/" + thestring):
    os.mkdir(output_path +"/" + thestring)

total_path = output_path + "/" +thestring
cur = time.time()
themean = 0.0
thelist=[]
for i in range(8000):
    print(i, "iteration")
    for j,raw_batch in enumerate(train_loader):

        print("iteration: ", i,"example in training set #: ", j)
        x_start = raw_batch['edited']
        raw_batch['edited'] = raw_batch['edited'].to(device)
        raw_batch['image'] = raw_batch['image'].to(device)
        newbatch = model.get_input(raw_batch, 'edited')
        raw_batch['caption'] = ['*']*raw_batch['image'].shape[0]
        z, cond = model.get_input(raw_batch, 'edited')
        #pdb.set_trace()
        samples, z_denoise_row = model.sample_log(cond=cond,batch_size=raw_batch['edited'].shape[0],ddim=True,ddim_steps=200,eta=1.0)
        decoded_samples = model.differentiable_decode_first_stage(samples).permute(0,2,3,1)
        optimizer.zero_grad()
        lossfun = torch.nn.MSELoss()
        loss = lossfun(decoded_samples, raw_batch['edited'])
        

        print("loss: ", loss)
        
        if not os.path.isdir(total_path + "/embeddings/"):
            os.mkdir(total_path + "/embeddings/")
        if not os.path.isdir(total_path + "/embeddings/" + thestring):
            os.mkdir(total_path + "/embeddings/" + thestring)
        model.embedding_manager.save(total_path + "/" +  "embeddings/" + thestring + "/embedding_" + str(i) + ".pt")
        
        if False: #i % 1 == 0 and j % 100 == 0:

            if not os.path.isdir(total_path + "/" + str(i)):
                os.mkdir(total_path + "/" + str(i))

            if not os.path.isdir(total_path + "/" + str(i) + "/" + str(j) + "/"):
                os.mkdir(total_path + "/" + str(i) + "/" + str(j) + "/")

            #pdb.set_trace()
            imgs = torch.clamp(raw_batch['image'].detach().cpu().permute(0,3,1,2), -1., 1)
            grid = torchvision.utils.make_grid(imgs, nrow=4)
            grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            im = Image.fromarray(grid)
            filename = str(loss.item()) + "_sample_" +"_inputs_" + str(j) + ".jpg"
            # pdb.set_trace()
            im.save(total_path + "/" + str(i) + "/" + str(j) + "/" + filename)

            imgs = torch.clamp(raw_eval_batch['image'].detach().cpu().permute(0,3,1,2), -1., 1)
            grid = torchvision.utils.make_grid(imgs, nrow=4)
            grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            im = Image.fromarray(grid)
            filename = str(loss.item()) + "_eval_" +"_inputs_" + str(j) + ".jpg"
            im.save(total_path + "/" + str(i) + "/" + str(j) + "/" + filename)

            log = model.log_images(7.5, 1.5, raw_batch)
            log_eval = model.log_images(7.5, 1.5, raw_eval_batch)
            #pdb.set_trace()
            key = 'samples'
            log[key] = torch.clamp(log[key].detach().cpu(), -1., 1)
            grid = torchvision.utils.make_grid(log[key], nrow=4)
            grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            im = Image.fromarray(grid)
            filename = str(loss.item()) + "_" + str(key) + "_" + str(j) + "_" + str(log["txt_scale"]) + \
                "_" + str(log["image_scale"]) + ".jpg"
            # pdb.set_trace()
            im.save(total_path + "/" + str(i) + "/" + str(j) + "/" + filename)
            # pdb.set_trace()
            

            log_eval[key] = torch.clamp(log_eval[key].detach().cpu(), -1., 1)
            grid = torchvision.utils.make_grid(log_eval[key], nrow=4)
            grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            im = Image.fromarray(grid)
            filename = str(loss.item()) + "_" + "eval" + "_" + str(j) + "_" + str(log["txt_scale"]) + \
                "_" + str(log["image_scale"]) + ".jpg"
            im.save(total_path + "/" + str(i) + "/" + str(j) + "/" + filename)
            #pdb.set_trace()
            
            
        #print(loss)
        
        
        thelist.append(loss.item())
        plt.figure()
        plt.plot(np.arange(0,len(thelist)),thelist)
        plt.savefig(output_path  + "/" + thestring + "/" + "currentloss.png")
        start_time = time.time()
        loss.backward()
        end_time = time.time()
        print(end_time-start_time, "backward")
        model.monitor_memory(5)
        start_time = time.time()
        optimizer.step()
        end_time = time.time()
        print(end_time-start_time, "step")
        #print(loss, "here")
        #pdb.set_trace()

        if model.use_scheduler:
            lr = model.optimizers().param_groups[0]['lr']
            model.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        new = time.time()

pdb.set_trace()

