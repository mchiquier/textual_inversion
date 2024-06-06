
#%%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#%%

import re
import time
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from ldm.util import instantiate_from_config
from ldm.data.personalized import  PersonalizedBase
from ldm.models.diffusion.ddpm_edit import LatentDiffusion
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
import numpy as np 
from PIL import Image
import os
import matplotlib.pyplot as plt
from clip import load as clip_load
import pdb
import clip
from torch.nn import functional as F
from PIL import Image
import PIL
from torch import autocast

import k_diffusion as K
from ldm.models.diffusion.ddpm_edit import CFGDenoiser
from einops import rearrange, repeat



transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class AFHQ(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder = folder_path
        self.transform = transform
        self.image_paths = os.listdir(folder_path)[:300]
        self.size = 256
        self.flip_p = 0.5
        # self.interpolation = {"linear": PIL.Image.LINEAR,
        #                       "bilinear": PIL.Image.BILINEAR,
        #                       "bicubic": PIL.Image.BICUBIC,
        #                       "lanczos": PIL.Image.LANCZOS,
        #                       }['bicubic']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        example = {}
        img_name = os.path.join(self.folder, self.image_paths[idx])
        image = Image.open(img_name)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = image.resize((self.size, self.size),PIL.Image.LANCZOS)

        #image = rearrange(2 * torch.tensor(np.array(image)).float() / 255 - 1, "h w c -> c h w")
        image = 2 * torch.tensor(np.array(image)).float() / 255 - 1
        
        #image = np.array(image).astype(np.uint8)
        example["image"] = image.permute(2,0,1) #torch.from_numpy((image / 127.5 - 1.0).astype(np.float32))
        example["caption"] = 'dog'
        example["file"] = img_name

        return example



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

def clean_string(input_string):
    # Remove any character that is not a letter
    cleaned_string = re.sub(r'[^a-zA-Z]', '_', input_string)
    # Convert to lowercase
    cleaned_string = cleaned_string.lower()
    return cleaned_string

'''
initialize model
'''
# load omega conf

class ImageBatch:

    def __init__(self,  model, img_size, device) -> None:

        self.device = device
        self.resize = transforms.Resize(img_size, interpolation=Image.BICUBIC)
        self.model = model 
        self.img_size = img_size
        self.temp = 0.07
    
    def encode_image(self,image):
        return F.normalize(self.model.encode_image(image.to(self.device)))
           

    def encode_text(self,attribute): 
        tokens = clip.tokenize(attribute).to(self.device)
        return F.normalize(self.model.encode_text(tokens))   
    
    def score(self, attr_encodings, img_encoding):

        with torch.no_grad():
            clsscores = (img_encoding @ attr_encodings)/self.temp
            cross_entropy_loss = torch.nn.CrossEntropyLoss()
            cross_entropy_score = cross_entropy_loss(clsscores, torch.tensor([1]).repeat(img_encoding.shape[0]).to(self.device)).item()
            return -cross_entropy_score, clsscores

def sample_old(model, input_images, edit, batch_size, ddim_steps):
    encoder_posterior_y = model.encode_first_stage(input_images)
    c = model.get_learned_conditioning([edit]*batch_size)
    print(len(c))
    xc = {}
    xc["c_crossattn"] = c[:batch_size]
    xc["c_concat"] = input_images[:batch_size]
    cond = {}
    uncond=0.05
    ddim_steps=200

    # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
    random = torch.rand(input_images.size(0), device=y.device)
    prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
    input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")

    null_prompt = model.get_learned_conditioning([""])
    cond["c_crossattn"] = [xc["c_crossattn"]]#[torch.where(prompt_mask, null_prompt, xc["c_crossattn"])] #.detach()
    cond["c_concat"] =  [input_mask * encoder_posterior_y.mode()] #[encoder_posterior_y.mode()] 

    torch.manual_seed(0)

    with model.ema_scope("Plotting"):
        samples, z_denoise_row = model.sample_log(cond=cond,batch_size=batch_size,ddim=False,ddim_steps=ddim_steps,eta=1.)
        #samples, z_denoise_row = model.sample(cond=cond, batch_size=batch_size, return_intermediates=True)
        x_samples = model.decode_first_stage(samples).permute(0,2,3,1)
    
    x_samples_decoded = ((x_samples+1)/2.0)*255 #between 0 and 255
    return x_samples_decoded

def euler_sampling(model, model_wrap, model_wrap_cfg,input_images, edit, batch_size, ddim_steps):
    list_of_edited_imgarray = []
    list_of_input_imgarray = []
    
    for i in range(input_images.shape[0]):
        

        input_image = input_images[i]
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            cond = {}
            input_image = input_image[None].to(model.device)
            cond["c_concat"] = [model.encode_first_stage(input_image).mode(),model.encode_first_stage(input_image).mode()]
            cond["c_crossattn"] = [model.get_learned_conditioning([edit])]
            
            null_token = model.get_learned_conditioning([""])

            uncond = {}
            uncond["c_crossattn"] = [null_token] 
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0]),torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(200)
            
            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": 7.5,
                "image_cfg_scale": 1.5,
            }
            torch.manual_seed(0)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            
            x = 255.0 * x.permute(0,2,3,1)#rearrange(x, "4 c h w -> 4 h w c")
            input_transformed = 255.0 * torch.clamp((input_image + 1.0) / 2.0, min=0.0, max=1.0).permute(0,2,3,1) #between 0 and 255
        
            list_of_edited_imgarray.append(x.cpu())
            list_of_input_imgarray.append(input_transformed.cpu())

    return torch.cat(list_of_edited_imgarray,dim=0), torch.cat(list_of_input_imgarray,dim=0)



if __name__ == '__main__': 

    if not os.path.isdir("results_mag"):
        os.mkdir("results_mag")

    if not os.path.isdir("results_mag/cat_to_dog"):
        os.mkdir("results_mag/cat_to_dog")

        
    dataset = AFHQ(folder_path='/proj/vondrick2/orr/projects/stargan-v2/data/afhq', transform=transform)

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = 0
    

    '''

    global vars, consider making a script to set these
    '''
    ckpt = 'models/ldm/stable-diffusion-v1/instruct-pix2pix-00-22000.ckpt'
    yml_paths = ['configs/latent-diffusion/instructpix2pix.yaml', ]

    config_paths = [OmegaConf.load(cfg) for cfg in yml_paths]
    dummy = OmegaConf.from_dotlist([])
    model_config = OmegaConf.merge(*config_paths, dummy)
    model_config.model.params.cond_stage_config.params.device=device
    train_config = model_config.pop("lightning", OmegaConf.create())
    # trainer_config = train_parameters.get("trainer", OmegaConf.create())
    model: LatentDiffusion = load_model_from_config(model_config, ckpt)
    model = model.eval().to(device)

    print("Size of dataset is: ", len(dataset))

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)

    hparams = {'image_size': 224, 'device': device, 'model_size': "ViT-B/32"}
    clip_model, preprocess = clip_load(hparams['model_size'], device=torch.device(hparams['device']), jit=False)  

    img_batch = ImageBatch(clip_model,224,torch.device(hparams['device']))
    cat_enc = img_batch.encode_text("a photo of a cat")
    dog_enc = img_batch.encode_text("a photo of a dog")
    encoded_classes_text = torch.cat([cat_enc, dog_enc],dim=0)
    edit = "add volume to the cat's cheeks" #"add whiskers to the cat"

    if not os.path.isdir("results_mag/cat_to_dog/" + clean_string(edit)):
        os.mkdir("results_mag/cat_to_dog/" + clean_string(edit))



    transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    list_of_class_scores = []
    list_of_x_sampled = []
    list_of_x_original = []




    ### EDITING + CLASSIFICATION PIPELINE ###

    for i,batch in enumerate(dataloader):
        timestart = time.time()
        print(i,len(dataloader))

        y = batch['image'].to(device)

        with torch.no_grad():

            ### EDIT IMAGES ###
            x_samples_decoded, input_images_list = euler_sampling(model, model_wrap, model_wrap_cfg, y, edit, batch_size, 200) #list of elements [1,256,256,3]

            ### TRANSFORM FOR CLIP ###
            resized_images=[]
            for iter in range(len(x_samples_decoded)):
                pil_image = Image.fromarray(x_samples_decoded[iter].cpu().numpy().astype(np.uint8))
                tensor_image = transform(pil_image.resize((224, 224),PIL.Image.LANCZOS))
                resized_images.append(tensor_image.float()[None])

            transformed_images = torch.cat(resized_images,dim=0)

            ### ENCODE EDITED IMAGES IN CLIP ###
            encoded_img = img_batch.encode_image(transformed_images)

            ### SCORE WITH CLIP ###
            crossentropyloss, class_scores = img_batch.score(encoded_classes_text.T, encoded_img)

            list_of_class_scores.append(class_scores)
            print(class_scores.shape)
            list_of_x_sampled.append(x_samples_decoded)
            list_of_x_original.append(input_images_list)
            #pdb.set_trace()
        timeend = time.time()
        print(timeend-timestart,i)
    
    all_class_scores = torch.cat(list_of_class_scores,dim=0)
    all_x_samples_decoded = torch.cat(list_of_x_sampled,dim=0)
    all_x_original = torch.cat(list_of_x_original,dim=0)
    result = all_class_scores[:, 1] > all_class_scores[:, 0]
    print((sum(result)/all_class_scores.shape[0]).item()*100)

    ### LOGGING ###
    torch.save(all_x_samples_decoded, "results_mag/cat_to_dog/" + clean_string(edit) + "/transformed.pt")
    torch.save(all_x_original, "results_mag/cat_to_dog/" + clean_string(edit) + "/original.pt")
    pdb.set_trace()

    with open("results_mag/cat_to_dog/" + clean_string(edit) + "/edit.txt", 'w') as file:
        file.write(edit + " \n")
        file.write(str((sum(result)/all_class_scores.shape[0]).item()*100) + " \n")
        converted_list = ["dog" if x==True else "cat" for x in result]
        for elem in converted_list:
            file.write(elem + " \n")
