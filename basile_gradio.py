'''
Created by Basile Van Hoorick, Jan 2024.
'''

import os  # noqa
import sys  # noqa
sys.path.insert(0, os.getcwd())  # noqa

# Library imports.
import argparse
import copy
import cv2
import fire
import gradio as gr
import glob
import joblib
import lovely_tensors
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pathlib
import random
import skimage
import skimage.metrics
import sys
import time
import torch
import torch.optim
import torchvision
import tqdm
import tqdm.rich
import warnings
from einops import rearrange
from functools import partial
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from tqdm import TqdmExperimentalWarning

# Internal imports.
import train_inversion

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

_TITLE = 'Visual Macros'

# This demo allows you to generate novel viewpoints of an object depicted in an input image using a fine-tuned version of Stable Diffusion.
_DESCRIPTION = '''
This demo represents a work in progress led by [Mia Chiquier](https://www.cs.columbia.edu/~mia.chiquier/) at [CVLab](https://www.cs.columbia.edu/~vondrick/) at Columbia University.

We apply textual inversion with a large-scale diffusion model to accomplish *low-shot image-to-image translation*.

In other words, we unlock the ability to approximate a wide range of visual tasks from just a few paired examples.

It is based on [InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix/).

Warning: All results are saved to the disk of the server for reproducibility and debugging purposes.
'''


def load_model_bundle(device):
    # NOTE: Adapted from train_inversion.py.

    ckpt = 'models/ldm/stable-diffusion-v1/instruct-pix2pix-00-22000.ckpt'
    yml_paths = ['configs/latent-diffusion/instructpix2pix.yaml']

    '''
    initialize model
    '''
    # load omega conf
    config_paths = [OmegaConf.load(cfg) for cfg in yml_paths]
    dummy = OmegaConf.from_dotlist([])
    model_config = OmegaConf.merge(*config_paths, dummy)
    train_config = model_config.pop("lightning", OmegaConf.create())
    # trainer_config = train_parameters.get("trainer", OmegaConf.create())

    model_bundle = [None, ckpt, device, model_config, train_config]
    return model_bundle


def main_run(model_bundle, cam_vis, output_path, action,
             input_file, edit_file, eval_file, center_crop,
             num_epochs, init_words, flip_aug, todo_aug,
             text_cfg, image_cfg, ddim_steps):
    # NOTE: Adapted from train_inversion.py.

    (model, ckpt, device, model_config, train_config) = model_bundle

    if action == 'train':
        # Always reload model to retrain / restart from scratch.
        model = train_inversion.load_model_from_config(model_config, ckpt)
        model = model.eval().to(device)
        model_bundle[0] = model
        pass

    # Get unique paths to store images to.
    dn_prefix = time.strftime('%Y%m%d-%H%M%S')
    input_a_dp = os.path.join(output_path, f'{dn_prefix}-in-a')
    input_b_dp = os.path.join(output_path, f'{dn_prefix}-in-b')
    input_eval_dp = os.path.join(output_path, f'{dn_prefix}-in-eval')
    output_train_dp = os.path.join(output_path, f'{dn_prefix}-out-a')
    output_eval_dp = os.path.join(output_path, f'{dn_prefix}-out-eval')

    # TODO save images to these dirs ^^

    '''
    create dataloader
    '''
    # Should be list. TODO verify
    model_config.model.params.personalization_config.params.initializer_words = \
        init_words.split(' ')

    model_config.data.params.train.params.data_root = input_a_dp
    model_config.data.params.validation.params.data_root = input_eval_dp
    model_config.data.params.train.params.edit_root = input_b_dp
    model_config.data.params.validation.params.edit_root = input_eval_dp

    # TODO support center crop etc
    data = train_inversion.instantiate_from_config(model_config.data)
    data.prepare_data()
    data.setup()
    train_loader = data._train_dataloader()
    val_loader = data._val_dataloader()

    raw_eval_batch = next(iter(val_loader))
    raw_eval_batch['edited'] = raw_eval_batch['edited'].to(device)
    raw_eval_batch['image'] = raw_eval_batch['image'].to(device)

    # TODO make this customizable!!
    raw_eval_batch['caption'] = ['*'] * raw_eval_batch['image'].shape[0]

    train_inversion.set_lr(model, model_config)
    embedding_params = list(model.embedding_manager.embedding_parameters())  # [self.parametersdiff]
    # model.embedding_manager.load("embeddings/purple/embedding_" + str(num) + ".pt")
    optimizer = torch.optim.AdamW(embedding_params, lr=model.learning_rate)

    cur = time.time()
    themean = 0.0
    thelist = []

    for i in range(6):
        print(i, "iteration")

        for j, raw_batch in enumerate(train_loader):
            print("iteration: ", i, "example in training set #: ", j)

            x_start = raw_batch['edited']
            raw_batch['edited'] = raw_batch['edited'].to(device)
            raw_batch['image'] = raw_batch['image'].to(device)
            # raw_batch['caption'] = ['make the statue purple']*raw_batch['image'].shape[0]
            newbatch = model.get_input(raw_batch, 'edited')
            # raw_batch['caption'] = ['make the statue *']*raw_batch['image'].shape[0]
            raw_batch['caption'] = ['*'] * raw_batch['image'].shape[0]
            batch = model.get_input(raw_batch, 'edited')

            optimizer.zero_grad()

            # calls p_losses
            # noise = default(noise, lambda: torch.randn_like(batch[0])).to(device)
            noise = torch.randn_like(batch[0]).to(device)
            t = torch.randint(0, 1000, (batch[0].shape[0],)).long().to('cuda')
            loss, loss_dict = model(batch[0], batch[1], t, noise)
            print("loss: ", loss)
            if not os.path.isdir(output_path + "/embeddings/"):
                os.mkdir(output_path + "/embeddings/")
            model.embedding_manager.save(
                output_path + "/" + "embeddings/" + "/embedding_" + str(i) + ".pt")

            if i % 1 == 0 and j % 100 == 0:

                if not os.path.isdir(output_path + "/" + str(i)):
                    os.mkdir(output_path + "/" + str(i))

                if not os.path.isdir(output_path + "/" + str(i) + "/" + str(j) + "/"):
                    os.mkdir(output_path + "/" + str(i) + "/" + str(j) + "/")

                log = model.log_images(7.5, 1.5, raw_batch)
                log_eval = model.log_images(7.5, 1.5, raw_eval_batch)
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
                im.save(output_path + "/" + str(i) + "/" + str(j) + "/" + filename)
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
                im.save(output_path + "/" + str(i) + "/" + str(j) + "/" + filename)
                # pdb.set_trace()

            # print(loss)

            thelist.append(loss.item())

            plt.figure()
            plt.plot(np.arange(0, len(thelist)), thelist)
            plt.savefig(output_path + "/" + "currentloss.png")
            plt.close()

            loss.backward()
            optimizer.step()
            # print(loss, "here")

            if model.use_scheduler:
                lr = model.optimizers().param_groups[0]['lr']
                model.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

            new = time.time()

    description = None
    train_gal = None
    test_gal = None

    if action == 'train':
        return (description, train_gal, test_gal)

    elif action == 'test':
        return (description, test_gal)


def run_demo(device='cuda',
             debug=True,
             output_path='../gradio_output_textinv/default'):

    model_bundle = load_model_bundle(device)

    os.makedirs(output_path, exist_ok=True)

    # Compose demo layout & data flow.
    demo = gr.Blocks(title=_TITLE)

    with demo:
        gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=0.9, variant='panel'):

                gr.Markdown('*First, upload training images here in the same order:*')
                input_file = gr.File(
                    file_count='multiple', file_types=['image'],
                    label='Input set A (clean images)')
                edit_file = gr.File(
                    file_count='multiple', file_types=['image'],
                    label='Input set B (edited images)')
                gr.Markdown('*Then, upload test images here:*')
                eval_file = gr.File(
                    file_count='multiple', file_types=['image'],
                    label='Eval set A (clean images)')

                gr.Markdown('*Processing options:*')
                center_crop_chk = gr.Checkbox(
                    True, label='Center crop to square aspect ratio')

                gr.Markdown('*Training options:*')
                epochs_sld = gr.Slider(
                    1, 20, value=6, step=1,
                    label='Number of epochs')
                words_sld = gr.Text(
                    'lol',
                    label='Initial words (that will be tokenized into the initial embeddings)')

                gr.Markdown('*Data augmentation options:*')
                flip_chk = gr.Checkbox(
                    True, label='Random horizontal flip')
                todo_chk = gr.Checkbox(
                    False, label='(todo implement more stuff later)')

                with gr.Accordion('Advanced options', open=False):
                    text_sld = gr.Slider(
                        1.0, 15.0, value=7.5, step=0.5,
                        label='Text CFG scale')
                    image_sld = gr.Slider(
                        1.0, 15.0, value=1.5, step=0.5,
                        label='Image CFG scale')
                    steps_sld = gr.Slider(
                        5, 200, value=100, step=5,
                        label='Number of DDIM steps')
                    # nosave_chk = gr.Checkbox(
                    #     False, label='Do not save generated video (only show in browser)')

                with gr.Row():
                    train_btn = gr.Button(
                        'Start Training', variant='primary')
                    test_btn = gr.Button(
                        'Evaluate Test Set', variant='primary')

                desc_output = gr.Markdown(
                    'First click Start Training. The results will appear on the right.')

            with gr.Column(scale=1.1, variant='panel'):

                # train_output = gr.Gallery(
                #     label='Train A to B')
                # train_output.style(grid=2)
                # test_output = gr.Gallery(
                #     label='Test A to B')
                # test_output.style(grid=2)

                train_output = gr.Image('Train A to B (each row is one epoch)')
                test_output = gr.Image('Test A to B (each row is one epoch)')

        train_btn.click(fn=partial(main_run, model_bundle, output_path, 'train'),
                        inputs=[input_file, edit_file, eval_file, center_crop_chk,
                                epochs_sld, words_sld, flip_chk, todo_chk,
                                text_sld, image_sld, steps_sld],
                        outputs=[desc_output, train_output, test_output],)

        test_btn.click(fn=partial(main_run, model_bundle, output_path, 'test'),
                       inputs=[input_file, edit_file, eval_file, center_crop_chk,
                               epochs_sld, words_sld, flip_chk, todo_chk,
                               text_sld, image_sld, steps_sld],
                       outputs=[desc_output, test_output],)

        gr.Markdown('Examples coming soon!')

    demo.launch(enable_queue=True, share=True, debug=debug)


if __name__ == '__main__':

    fire.Fire(run_demo)

    pass
