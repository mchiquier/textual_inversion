# BVH, Jan 2024.

# https://github.com/mchiquier/textual_inversion

cdb4 && cd VLC4D/textual_inversion
ma p310cu118


# install required stuff:

pip install git+https://github.com/crowsonkb/k-diffusion.git

# pip install git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
# ^ cannot find taming
# pip install taming-transformers
# ^ still cannot find taming
# pip install taming-transformers-rom1504
# ^ now kind of works but main.DataModuleFromConfig is problematic

pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers

pip install -e .

# try example commmand:

export DATA_ROOT="data/headshot"
export EDIT_ROOT="data/headshothat"
export EVAL_ROOT="data/headshoteval"
export OUTPUT_PATH="results/trilbyhat"

python train_inversion.py --data_root=$DATA_ROOT \
                          --edit_root=$EDIT_ROOT \
                          --eval_root=$EVAL_ROOT \
                          --output_path=$OUTPUT_PATH \
                          --init_words "word1" "word2"
