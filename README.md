# Visual Macros

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2208.01618-b31b1b.svg)](https://arxiv.org/abs/2208.01618)

[[Project Website](https://textual-inversion.github.io/)]

> **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion**<br>
> Rinon Gal<sup>1,2</sup>, Yuval Alaluf<sup>1</sup>, Yuval Atzmon<sup>2</sup>, Or Patashnik<sup>1</sup>, Amit H. Bermano<sup>1</sup>, Gal Chechik<sup>2</sup>, Daniel Cohen-Or<sup>1</sup> <br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>NVIDIA

>**Abstract**: <br>
> Text-to-image models offer unprecedented freedom to guide creation through natural language.
  Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes.
  In other words, we ask: how can we use language-guided models to turn <i>our</i> cat into a painting, or imagine a new product based on <i>our</i> favorite toy?
  Here we present a simple approach that allows such creative freedom.
  Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model.
  These "words" can be composed into natural language sentences, guiding <i>personalized</i> creation in an intuitive way.
  Notably, we find evidence that a <i>single</i> word embedding is sufficient for capturing unique and varied concepts.
  We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks. -->

## Description
This repo is a wild mix of InstructPix2Pix and textual invesion. 

<!-- ## Updates
**29/08/2022** Merge embeddings now supports SD embeddings. Added SD pivotal tuning code (WIP), fixed training duration, checkpoint save iterations.
**21/08/2022** Code released!

## TODO:
- [x] Release code!
- [x] Optimize gradient storing / checkpointing. Memory requirements, training times reduced by ~55%
- [x] Release data sets
- [ ] Release pre-trained embeddings
- [ ] Add Stable Diffusion support -->

## Setup

<!-- Our code builds on, and shares requirements with [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion). To set up their environment, please run: -->
Setup using pip:
```
pip install -r requirements.txt
```
Setup using conda:
```
conda env create -f environment.yaml
conda activate ldm
```

You will also need the Instructpix2pix model checkpoint. 

Currently, the model can be downloaded by running:

```
mkdir -p models/ldm/stable-diffusion-v1/
wget -O models/ldm/stable-diffusion-v1/instruct-pix2pix-00-22000.ckpt http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt
```

## Usage

### Inversion

To invert an image set, run:

```
export DATA_ROOT="data/headshot"
export EDIT_ROOT="data/headshothat"
export EVAL_ROOT="data/headshoteval"
export OUTPUT_PATH="results/trilbyhat"

python train_inversion.py --data_root=$DATA_ROOT \
                          --edit_root=$EDIT_ROOT \
                          --eval_root=$EVAL_ROOT \
                          --output_path=$OUTPUT_PATH \
                          --init_words "word1" "word2"

```
