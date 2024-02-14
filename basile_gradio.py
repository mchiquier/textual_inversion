'''
Created by Basile Van Hoorick, Jan 2024.

cdb4 && cd VLC4D/textual_inversion
ma p310cu118
CUDA_VISIBLE_DEVICES=1 python basile_gradio.py --port=7881 \
--output_path=../gradio_output_textinv/internal
CUDA_VISIBLE_DEVICES=2 python basile_gradio.py --port=7882 \
--output_path=../gradio_output_textinv/internal
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
import shutil
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

Make sure to try "Visualize Data" first to verify whether the demo is responsive.

Warning: All results are saved to the disk of the server for reproducibility and debugging purposes.
'''

os.environ['GRADIO_TEMP_DIR'] = f'/tmp/gradio_{np.random.randint(1000, 10000)}'


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


def main_run(model_bundle, output_path, action,
             root_drop, root_text,
             input_file, edit_file, eval_file,
             which_one, center_crop,
             which_task, num_epochs, init_words,
             flip_aug, crop_aug, blur_aug, noise_aug,
             vectors_pt, text_cfg, image_cfg, ddim_steps):
    '''
    :param input_file: List of tempfile.TemporaryFileWrapper objects.
    :param edit_file: List of tempfile.TemporaryFileWrapper objects.
    :param eval_file: List of tempfile.TemporaryFileWrapper objects.
    '''
    # NOTE: Adapted from train_inversion.py.
    (model, ckpt, device, model_config, train_config) = model_bundle

    # Get unique paths to store images to.
    dn_prefix = time.strftime('%Y%m%d-%H%M%S')
    raw_train_a_dp = os.path.join(output_path, dn_prefix, 'raw-train-a')
    raw_train_b_dp = os.path.join(output_path, dn_prefix, 'raw-train-b')
    raw_eval_a_dp = os.path.join(output_path, dn_prefix, 'raw-eval-a')
    in_train_a_dp = os.path.join(output_path, dn_prefix, 'in-train-a')
    in_train_b_dp = os.path.join(output_path, dn_prefix, 'in-train-b')
    in_eval_a_dp = os.path.join(output_path, dn_prefix, 'in-eval-a')
    out_train_dp = os.path.join(output_path, dn_prefix, 'out-train')
    out_eval_dp = os.path.join(output_path, dn_prefix, 'out-eval')
    vis_dp = os.path.join(output_path, dn_prefix, 'vis')
    embeddings_dp = os.path.join(output_path, dn_prefix, 'embeddings')

    os.makedirs(raw_train_a_dp, exist_ok=True)
    os.makedirs(raw_train_b_dp, exist_ok=True)
    os.makedirs(raw_eval_a_dp, exist_ok=True)
    os.makedirs(in_train_a_dp, exist_ok=True)
    os.makedirs(in_train_b_dp, exist_ok=True)
    os.makedirs(in_eval_a_dp, exist_ok=True)
    os.makedirs(out_train_dp, exist_ok=True)
    os.makedirs(out_eval_dp, exist_ok=True)
    os.makedirs(vis_dp, exist_ok=True)
    os.makedirs(embeddings_dp, exist_ok=True)

    # Copy image files.
    if 'upload' in which_one.lower():
        for (i, tmp_file) in enumerate(input_file):
            dst_fp = os.path.join(
                raw_train_a_dp, f'clean_{i:03d}.{os.path.splitext(tmp_file.name)[1]}')
            shutil.copy(tmp_file.name, dst_fp)
        for (i, tmp_file) in enumerate(edit_file):
            dst_fp = os.path.join(
                raw_train_b_dp, f'edit_{i:03d}.{os.path.splitext(tmp_file.name)[1]}')
            shutil.copy(tmp_file.name, dst_fp)
        for (i, tmp_file) in enumerate(eval_file):
            dst_fp = os.path.join(
                raw_eval_a_dp, f'eval_{i:03d}.{os.path.splitext(tmp_file.name)[1]}')
            shutil.copy(tmp_file.name, dst_fp)

    else:
        if 'dropdown' in which_one.lower():
            root_path = root_drop
        elif 'text' in which_one.lower():
            root_path = root_text
        print(f'Using root_path: {root_path}')

        raw_train_a_fps = glob.glob(os.path.join(root_path, 'train-a', '*.*'))
        raw_train_b_fps = glob.glob(os.path.join(root_path, 'train-b', '*.*'))
        raw_eval_a_fps = glob.glob(os.path.join(root_path, 'eval-a', '*.*'))
        print(f'Found {len(raw_train_a_fps)} train-a images.')
        print(f'Found {len(raw_train_b_fps)} train-b images.')
        print(f'Found {len(raw_eval_a_fps)} eval-a images.')

        if 'map' in which_task.lower():
            if len(raw_train_a_fps) == 0 or len(raw_train_b_fps) == 0 or len(raw_eval_a_fps) == 0:
                raise ValueError('No images found in at least one subfolder.')
            if len(raw_train_a_fps) != len(raw_train_b_fps):
                raise ValueError('Train-a and train-b must have the same number of images.')
            if len(raw_train_a_fps) != len(raw_eval_a_fps):
                raise ValueError('Train-a and eval-a must have the same number of images.')

        else:
            # Procedural ground truth.
            if len(raw_train_a_fps) == 0 or len(raw_eval_a_fps) == 0:
                raise ValueError('No images found in at least one subfolder.')
            if len(raw_train_a_fps) != len(raw_eval_a_fps):
                raise ValueError('Train-a and eval-a must have the same number of images.')

        for (i, fp) in enumerate(raw_train_a_fps):
            dst_fp = os.path.join(
                raw_train_a_dp, f'clean_{i:03d}.{os.path.splitext(fp)[1]}')
            shutil.copy(fp, dst_fp)
        for (i, fp) in enumerate(raw_train_b_fps):
            dst_fp = os.path.join(
                raw_train_b_dp, f'edit_{i:03d}.{os.path.splitext(fp)[1]}')
            shutil.copy(fp, dst_fp)
        for (i, fp) in enumerate(raw_eval_a_fps):
            dst_fp = os.path.join(
                raw_eval_a_dp, f'eval_{i:03d}.{os.path.splitext(fp)[1]}')
            shutil.copy(fp, dst_fp)

    '''
    create dataloader
    '''
    # Assign dataset options.
    model_config.data.params.train.params.data_root = raw_train_a_dp
    model_config.data.params.validation.params.data_root = raw_eval_a_dp
    model_config.data.params.train.params.edit_root = raw_train_b_dp
    model_config.data.params.validation.params.edit_root = raw_eval_a_dp
    model_config.data.params.train.params.size = 256
    model_config.data.params.validation.params.size = 256
    model_config.data.params.train.params.center_crop = center_crop
    model_config.data.params.validation.params.center_crop = center_crop
    model_config.data.params.train.params.flip_p = 0.5 if flip_aug else 0.0
    model_config.data.params.train.params.horizontal_flip=True if flip_aug else False
    model_config.data.params.train.params.random_crop=True if crop_aug  else False
    model_config.data.params.train.params.gaussian_blur=True if blur_aug else False 
    model_config.data.params.train.params.gaussian_noise=True if noise_aug else False
    model_config.data.params.validation.params.flip_p = 0.0
    model_config.data.params.train.params.crop_p = 0.8 if crop_aug else 0.0
    model_config.data.params.validation.params.crop_p = 0.0
    model_config.data.params.train.params.procedural_task = 'ab'  # TODO

    data = train_inversion.instantiate_from_config(model_config.data)
    data.prepare_data()
    data.setup()
    train_loader = data._train_dataloader()
    val_loader = data._val_dataloader()

    raw_eval_batch = next(iter(val_loader))
    raw_eval_batch['edited'] = raw_eval_batch['edited'].to(device)
    raw_eval_batch['image'] = raw_eval_batch['image'].to(device)

    # TODO: make this customizable!
    raw_eval_batch['caption'] = ['*'] * raw_eval_batch['image'].shape[0]

    # NOTE: There is some overlap with the data loading code here.
    my_tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256)])

    imgs_train_a = [my_tf(Image.open(fp)) for fp in sorted(glob.glob(raw_train_a_dp + '/*.*'))]
    imgs_train_b = [my_tf(Image.open(fp)) for fp in sorted(glob.glob(raw_train_b_dp + '/*.*'))]
    imgs_eval_a = [my_tf(Image.open(fp)) for fp in sorted(glob.glob(raw_eval_a_dp + '/*.*'))]
    data_gal = rearrange(torch.stack(
        [torch.stack(imgs_train_a), torch.stack(imgs_train_b), torch.stack(imgs_eval_a)], dim=0),
        'K R C H W -> C (K H) (R W)')
    data_gal = (data_gal * 255).numpy().astype(np.uint8)
    data_gal = rearrange(data_gal, 'C H W -> H W C')
    # TODO ^ save this to input image dirs too

    plt.imsave(os.path.join(vis_dp, 'data.jpg'), data_gal)

    if 'data' in action:
        description = ('The input images are visualized on the right. '
                       'If this looks correct, you may proceed to Train & Evaluate.')
        # return (description, data_gal, None, None)
        return (description, data_gal)

    '''
    initialize model
    '''
    # Should be list. TODO verify
    model_config.model.params.personalization_config.params.initializer_words = \
        init_words.split(' ')
    model_config.model.params.personalization_config.params.num_vectors_per_token = \
        vectors_pt

    if 'train' in action:
        # Always reload model to retrain / restart from scratch.
        model = train_inversion.load_model_from_config(model_config, ckpt)
        model = model.eval().to(device)
        model_bundle[0] = model
        pass

    train_inversion.set_lr(model, model_config)
    embedding_params = list(model.embedding_manager.embedding_parameters())
    optimizer = torch.optim.AdamW(embedding_params, lr=model.learning_rate)

    thelist = []
    train_gal = []
    test_gal = []

    for i in tqdm.tqdm(range(num_epochs)):
        print()
        print(f'[cyan][Start Epoch {i + 1} / {num_epochs}]')

        for (j, raw_batch) in enumerate(train_loader):

            raw_batch['image'] = raw_batch['image'].to(device)
            raw_batch['edited'] = raw_batch['edited'].to(device)
            # raw_batch['caption'] = ['make the statue purple']*raw_batch['image'].shape[0]
            # raw_batch['caption'] = ['make the statue *']*raw_batch['image'].shape[0]

            # TODO make this customizable!
            raw_batch['caption'] = ['*'] * raw_batch['image'].shape[0]
            batch = model.get_input(raw_batch, 'edited')

            optimizer.zero_grad()

            # calls p_losses
            # noise = default(noise, lambda: torch.randn_like(batch[0])).to(device)
            noise = torch.randn_like(batch[0]).to(device)
            t = torch.randint(0, 1000, (batch[0].shape[0],)).long().to('cuda')
            (loss, loss_dict) = model(batch[0], batch[1], t, noise)

            thelist.append(loss.item())
            loss.backward()
            optimizer.step()

            if model.use_scheduler:
                lr = model.optimizers().param_groups[0]['lr']
                model.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

            if j % 10 == 0:
                print(f'[gray][E {i + 1} / {num_epochs} S {j + 1} / {len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')

            if j % 20 == 0:
                emb_fp = os.path.join(embeddings_dp, f'embedding_{i:03d}_{j:03d}.pt')
                model.embedding_manager.save(emb_fp)

            # Sample and save stuff before every epoch, but also after the last one.
            if j == 0 or (i == num_epochs - 1 and j == len(train_loader) - 1):
                BL = 8  # For efficiency.
                limit_batch = raw_batch
                limit_batch['image'] = limit_batch['image'][:BL]
                limit_batch['edited'] = limit_batch['edited'][:BL]
                limit_batch['caption'] = limit_batch['caption'][:BL]

                print(f'Sampling with train set images...')
                log = model.log_images(
                    text_cfg, image_cfg, raw_batch, ddim_steps=ddim_steps)

                print(f'Sampling with test set images...')
                log_eval = model.log_images(
                    text_cfg, image_cfg, raw_eval_batch, ddim_steps=ddim_steps)

                # NOTE: log and log_eval are both dicts with keys:
                # inputs, reals. reconstruction, conditioning, txt_scale, img_scale, samples.

                key = 'samples'
                log[key] = torch.clamp(log[key].detach().cpu(), -1., 1)
                grid = torchvision.utils.make_grid(log[key], nrow=4)
                grid = (grid + 1.0) / 2.0
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)

                # im = Image.fromarray(grid)
                # filename = str(loss.item()) + "_" + str(key) + "_" + str(j) + "_" + str(log["txt_scale"]) + \
                #     "_" + str(log["image_scale"]) + ".jpg"
                # im.save(output_path + "/" + str(i) + "/" + str(j) + "/" + filename)

                train_gal.append(grid)
                plt.imsave(os.path.join(vis_dp, f'train_{i:03d}_{j:03d}.jpg'), grid)

                log_eval[key] = torch.clamp(log_eval[key].detach().cpu(), -1., 1)
                grid = torchvision.utils.make_grid(log_eval[key], nrow=4)
                grid = (grid + 1.0) / 2.0
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)

                # im = Image.fromarray(grid)
                # filename = str(loss.item()) + "_" + "eval" + "_" + str(j) + "_" + str(log["txt_scale"]) + \
                #     "_" + str(log["image_scale"]) + ".jpg"
                # im.save(output_path + "/" + str(i) + "/" + str(j) + "/" + filename)

                test_gal.append(grid)
                plt.imsave(os.path.join(vis_dp, f'test_{i:03d}_{j:03d}.jpg'), grid)

                print(f'[yellow]Saved visuals')
                print(f'[yellow]Current time: {time.strftime("%Y%m%d-%H%M%S")}')

                plt.figure()
                plt.plot(np.arange(0, len(thelist)), thelist)
                plt.savefig(os.path.join(vis_dp, 'loss.png'))
                plt.close()

    description = 'Done!'

    train_gal = rearrange(np.stack(train_gal), 'K H W C -> (K H) W C')
    plt.imsave(os.path.join(vis_dp, 'train.jpg'), train_gal)

    test_gal = rearrange(np.stack(test_gal), 'K H W C -> (K H) W C')
    plt.imsave(os.path.join(vis_dp, 'test.jpg'), test_gal)

    loss_vis = plt.imread(os.path.join(vis_dp, 'loss.png'))

    # return (description, data_gal, train_gal, test_gal)
    return (description, train_gal, test_gal, loss_vis)


def run_demo(device='cuda',
             debug=True,
             output_path='../gradio_output_textinv/default',
             port=7870):

    model_bundle = load_model_bundle(device)

    os.makedirs(output_path, exist_ok=True)

    # Compose demo layout & data flow.
    demo = gr.Blocks(title=_TITLE)

    with demo:
        gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=0.9, variant='panel'):

                gr.Markdown('*Select path to root data folder:*')
                root_choices = [
                    '/proj/vondrick4/VisualMacros/Photoshop/hat',
                    '/proj/vondrick4/VisualMacros/Photoshop/reddot',
                    '/proj/vondrick4/VisualMacros/NovelView/backpack',
                    '/proj/vondrick4/VisualMacros/NovelView/car',
                ]
                root_drop = gr.Dropdown(
                    root_choices,
                    value='/proj/vondrick4/VisualMacros/NovelView/backpack',
                    label='Path (must contain train-a, train-b, eval-a)'
                )
                refresh_btn = gr.Button(
                    'Discover Datasets', variant='secondary')

                gr.Markdown('*Alternatively, type direct path to root data folder:*')
                root_text = gr.Text(
                    label='Path (must contain train-a, train-b, eval-a)')

                gr.Markdown('*Alternatively, upload training images here in the same order:*')
                input_file = gr.File(
                    file_count='multiple', file_types=['image'],
                    label='Train set A (clean images)')
                edit_file = gr.File(
                    file_count='multiple', file_types=['image'],
                    label='Train set B (edited images)')
                gr.Markdown('*Then, upload test images here:*')
                eval_file = gr.File(
                    file_count='multiple', file_types=['image'],
                    label='Eval set A (clean images)')

                gr.Markdown('*Processing options:*')
                which_rad = gr.Radio(
                    ['Dropdown path', 'Text path', 'Uploaded files'],
                    value='Dropdown path',
                    label='Select which input to train and test the model with')
                center_crop_chk = gr.Checkbox(
                    True, label='Center crop to square aspect ratio')

                gr.Markdown('*Training options:*')
                task_rad = gr.Radio(
                    ['Map A to B', 'Flip A', 'Invert A'],
                    value='Map A to B',
                    label='How to supervise the model (todo not yet implemented)')
                epochs_sld = gr.Slider(
                    1, 20, value=6, step=1,
                    label='Number of epochs')
                words_sld = gr.Text(
                    'word lol',
                    label='Initial words (that will be tokenized into the initial embeddings)')

                gr.Markdown('*Data augmentation options:*')
                flip_chk = gr.Checkbox(
                    True, label='Random horizontal flip')
                crop_chk = gr.Checkbox(
                    False, label='Random crop')
                blur_chk = gr.Checkbox(
                    False, label='Gaussian blur')
                noise_chk = gr.Checkbox(
                    False, label='Gaussian noise')
                

                with gr.Accordion('Advanced options', open=False):
                    vectors_sld = gr.Slider(
                        1, 20, value=10, step=1,
                        label='Number of vectors per token')
                    text_sld = gr.Slider(
                        1.0, 15.0, value=7.5, step=0.5,
                        label='Text CFG scale')
                    image_sld = gr.Slider(
                        1.0, 15.0, value=1.5, step=0.5,
                        label='Image CFG scale')
                    steps_sld = gr.Slider(
                        5, 200, value=50, step=5,
                        label='Number of DDIM steps')
                    # nosave_chk = gr.Checkbox(
                    #     False, label='Do not save generated video (only show in browser)')

                with gr.Row():
                    data_btn = gr.Button(
                        'Visualize Data', variant='secondary')
                    # train_btn = gr.Button(
                    #     'Start Training', variant='primary')
                    # test_btn = gr.Button(
                    #     'Evaluate Test Set', variant='primary')
                    train_btn = gr.Button(
                        'Train & Evaluate', variant='primary')

                desc_output = gr.Markdown(
                    'The results will appear on the right.')

            with gr.Column(scale=1.1, variant='panel'):

                # train_output = gr.Gallery(
                #     label='Train A to B')
                # train_output.style(grid=2)
                # test_output = gr.Gallery(
                #     label='Test A to B')
                # test_output.style(grid=2)

                data_output = gr.Image(
                    label='Dataset visualization (rows = A | B | eval)')
                train_output = gr.Image(
                    label='Train A to B (epochs are stacked vertically)')
                test_output = gr.Image(
                    label='Test A to B (epochs are stacked vertically)')
                loss_output = gr.Image(
                    label='Training loss curve')

        def refresh_datasets():
            # https://github.com/gradio-app/gradio/issues/6862#issuecomment-1866577714
            root_choices = glob.glob(r'/proj/vondrick4/VisualMacros/*/*')
            # glob.glob(r'/proj/vondrick4/VisualMacros/*')
            print('root_choices:', root_choices)
            # root_choices = [str(pathlib.Path(dp).parent) for dp in root_choices]
            root_drop = gr.Dropdown(
                root_choices,
                value=root_choices[0],
                label='Path (must contain train-a, train-b, eval-a)',
                interactive=True,
            )
            return root_drop

        refresh_btn.click(fn=refresh_datasets, outputs=[root_drop])

        data_btn.click(fn=partial(main_run, model_bundle, output_path, 'data'),
                       inputs=[root_drop, root_text,
                               input_file, edit_file, eval_file,
                               which_rad, center_crop_chk,
                               task_rad, epochs_sld, words_sld,
                               flip_chk, crop_chk, blur_chk, noise_chk,
                               vectors_sld, text_sld, image_sld, steps_sld],
                       outputs=[desc_output, data_output])

        train_btn.click(fn=partial(main_run, model_bundle, output_path, 'train'),
                        inputs=[root_drop, root_text,
                                input_file, edit_file, eval_file,
                                which_rad, center_crop_chk,
                                task_rad, epochs_sld, words_sld,
                                flip_chk, crop_chk, blur_chk, noise_chk, 
                                vectors_sld, text_sld, image_sld, steps_sld],
                        outputs=[desc_output, train_output, test_output, loss_output])

        # test_btn.click(fn=partial(main_run, model_bundle, output_path, 'test'),
        #                inputs=[input_file, edit_file, eval_file, center_crop_chk,
        #                        epochs_sld, words_sld, flip_chk, todo_chk,
        #                        text_sld, image_sld, steps_sld],
        #                outputs=[desc_output, test_output],)

        gr.Markdown('Examples coming soon!')

    demo.queue(max_size=20)
    demo.launch(share=True, debug=debug, server_port=port)


if __name__ == '__main__':

    fire.Fire(run_demo)

    pass
