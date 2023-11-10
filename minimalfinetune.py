import torch
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
'''

global vars, consider making a script to set these
'''
ckpt = 'models/ldm/stable-diffusion-v1/instruct-pix2pix-00-22000.ckpt'
yml_paths = ['configs/latent-diffusion/instructpix2pix.yaml', ]
data_root = 'destdir'
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
embedding_params = list(model.model.diffusion_model.parameters())#[self.parametersdiff]
num=100
#model.embedding_manager.load('embeddings/inference_noperimgtoken_dropout_05_bs16_juststar_10vectors_cloudsstar_finetune/embedding_0_480.ckpt')

#model.embedding_manager.load("embeddings/purple/embedding_" + str(num) + ".pt")
optimizer = torch.optim.AdamW(embedding_params, lr=model.learning_rate)
raw_batch = next(iter(train_loader))
#torch.manual_seed(0)
noise=None
raw_batch['edited'] = raw_batch['edited'].to(device)
raw_batch['image'] = raw_batch['image'].to(device)
raw_batch['caption'] = ['*', '*', '*', '*','*', '*', '*', '*','*', '*', '*', '*','*', '*', '*', '*']
thestring="inference/inference_noperimgtoken_dropout_05_bs16_juststar_10vectors_cloudsstar_finetune_two"
stringforcheckpoint = thestring.split("/")[-1]
batch = model.get_input(raw_batch, 'edited')
if not os.path.isdir(thestring):
    os.mkdir(thestring)
cur = time.time()
themean = 0.0
thelist=[]
for i in range(8000):
    print(i, "here")
    #pdb.set_trace()
    for j,raw_batch in enumerate(train_loader):
        print(i,j)
        x_start = raw_batch['edited']
        raw_batch['edited'] = raw_batch['edited'].to(device)
        raw_batch['image'] = raw_batch['image'].to(device)
        raw_batch['caption'] = ['make the statue purple', 'make the statue purple', 'make the statue purple', 'make the statue purple','make the statue purple', 'make the statue purple', 'make the statue purple', 'make the statue purple',
                                'make the statue purple', 'make the statue purple', 'make the statue purple', 'make the statue purple','make the statue purple', 'make the statue purple', 'make the statue purple', 'make the statue purple']
        newbatch = model.get_input(raw_batch, 'edited')
        raw_batch['caption'] = ['add * to the image']*16
        #raw_batch['caption'] = ['*', '*', '*', '*','*', '*', '*', '*','*', '*', '*', '*','*', '*', '*', '*']
        batch = model.get_input(raw_batch, 'edited')
        
        #batch[1]['c_crossattn'][0] = newbatch[1]['c_crossattn'][0] 
        #pdb.set_trace()
        # NOTE: way it is currently coded the k='edited' parameter does not do anything..

        optimizer.zero_grad()

        lossfun = torch.nn.MSELoss()
        #pdb.set_trace()
        #loss = lossfun(batch[1]['c_crossattn'][0],newbatch[1]['c_crossattn'][0])
        #model.zero_grad()
        #print(loss)


        # calls p_losses
        noise = default(noise, lambda: torch.randn_like(batch[0])).to(device)
        t = torch.randint(0, 1000, (batch[0].shape[0],)).long().to('cuda')
        loss, loss_dict = model(batch[0], batch[1],t, noise)
        print(loss)
        #print("hi")
        if i%1==0 and j%1==0:
            out = raw_batch['edited'].detach().cpu().permute(0,3,1,2)
            grid = torchvision.utils.make_grid(out, nrow=4)
            grid = (grid + 1.0) / 2.0 
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            #pdb.set_trace()
            im = Image.fromarray(grid)
            filename = 'rawedited' + str(i) + ".jpg"
            

            if not os.path.isdir(thestring + "/" + str(i)):
                os.mkdir(thestring + "/" + str(i))

            if not os.path.isdir(thestring + "/" + str(i) + "/" + str(j) + "/"):
                os.mkdir(thestring + "/" + str(i) + "/" +str(j) + "/")
    
            im.save(thestring + "/" + str(i) + "/" + str(j) + "/" + filename)
            log = model.log_images(7.5, 1.5, raw_batch)
            for key in log.keys():
                if key =='samples':
                    log[key] = torch.clamp(log[key].detach().cpu(),-1.,1)
                    grid = torchvision.utils.make_grid(log[key], nrow=4)
                    grid = (grid + 1.0) / 2.0 
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid * 255).astype(np.uint8)
                    im = Image.fromarray(grid)
                    filename = str(loss.item()) + "_" + str(key) + "_" + str(j) + "_" + str(log["txt_scale"]) + \
                    "_" + str(log["image_scale"]) + ".jpg"
                    #pdb.set_trace()
                    im.save(thestring + "/" + str(i) + "/" + str(j) + "/" + filename)
                    #pdb.set_trace()
                if key =='reals':
                    log[key] = torch.clamp(log[key].detach().cpu(),-1.,1)
                    grid = torchvision.utils.make_grid(log[key], nrow=4)
                    grid = (grid + 1.0) / 2.0 
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid * 255).astype(np.uint8)
                    im = Image.fromarray(grid)
                    filename = str(loss.item()) + "_" + str(key) + "_" + str(j) + "_" + str(log["txt_scale"]) + \
                    "_" + str(log["image_scale"]) + ".jpg"
                    im.save(thestring + "/" + str(i) + "/" + str(j) + "/" + filename)
            #pdb.set_trace()
            if j==480:
                if not os.path.isdir("pretrainedmodels/" + stringforcheckpoint):
                    os.mkdir("pretrainedmodels/" + stringforcheckpoint)
                model.embedding_manager.save("pretrainedmodels/" + stringforcheckpoint + "/model_" + str(i) + "_" + str(j) + ".pt")
            
        #print(loss)
        
        if i%10==0:
            if i!=0 :
                print(torch.mean(torch.tensor(thelist)))
            thelist=[]
        else:
            thelist.append(loss)
        loss.backward()
        optimizer.step()
        #print(loss, "here")

        if model.use_scheduler:
            lr = model.optimizers().param_groups[0]['lr']
            model.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        new = time.time()
    #pdb.set_trace()
    #torch.save(optimizer.param_groups[0]['params'][0], "current_embed.ckpt")
    #print(cur-new)
    # # NOTE: probably does not matter as model not updated
    # if model.use_ema:
    #     model.model_ema(model)
pdb.set_trace()

