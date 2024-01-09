# IBP Regularization for Verified Adversarial Robustness via Branch-and-Bound

This repository provides code for [IBP Regularization for Verified Adversarial Robustness via Branch-and-Bound](https://arxiv.org/abs/2206.14772).
If you use it in your research, please cite:
```
@Article{DePalma2022,
        author={De Palma, Alessandro and Bunel, Rudy and Dvijotham, Krishnamurthy and Kumar, M. Pawan and Stanforth, Robert},
        booktitle={ICML 2022 Workshop on Formal Verification of Machine Learning},
        title={{IBP} Regularization for Verified Adversarial Robustness via Branch-and-Bound},
        volume={abs/2206.14772},
        year={2022},
}
```

## Setup

We recommend using a Conda enviroment:
```
conda create -y -n ibpr python=3.7
conda activate ibpr
```

### Dependencies

The code requires [Jax](https://github.com/google/jax) version 0.2.27 (with  `jaxlib` version 0.1.75),
[PyTorch](https://pytorch.org/get-started/) version 1.11.0, and TensorFlow version 2.10.
We recommend installing them as follows:
```
# Jax, with CUDA support, needs to be installed from source
# Note that jax versions after 0.2.27 are not tested
conda install -c anaconda cudnn=8.2.1 cudatoolkit=11.3
pip install --upgrade jaxlib==0.1.75 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade "jax[cuda11_pip]==0.2.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Check PyTorch's website for alternative installation instructions.
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Tensorflow (2.10) is required to run the verification code (specifically, for the jax to torch conversion)
pip install tensorflow==2.10.*
```

The training pipeline requires `haiku` and `jax_verify`, to be installed as follows:
```
pip install dm-haiku==0.0.5
git clone https://github.com/deepmind/jax_verify.git
cd jax_verify; git checkout a9e8cac9a80fdcdc030a0b4dc7a759f654104cc4
vim requirements.txt +9d +x; vim requirements.txt +4d +x; vim requirements.txt +4d +x; vim requirements.txt +5d +x; echo "einshape" >> requirements.txt; echo "optax==0.1.0" >> requirements.txt  # see https://github.com/deepmind/jax_verify/issues/8
pip install .; cd ..
```

In order to execute the verification code, the [OVAL](https://github.com/oval-group/oval-bab/) verification framework
should be downloaded and installed by following [these instructions](https://github.com/oval-group/oval-bab/blob/main/README.md).

### Installation

After complying with the above, the code of this repository can be installed by running
`pip install .` (the remaining dependencies are listed in `setup.py`).

## Training and Verification

Example pre-trained models are provided in the `saved_models` directory.

### CIFAR-10 - 2/255
```
# train
python train/train_main.py --train_mode train --dataset cifar10 --net convmedbig_flat_2_2_4_250 --train_batch 100 --test_batch 100 --train_eps 0.00784313725 --start_eps_factor 2.1 --anneal --train_att_n_steps 8 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.035 --opt sgd --lr 1e-2 --lr_factor 0.95 --cont_lr_decay --cont_lr_mix --mix --mix_epochs 600 --n_epochs 800 --l1_reg 0.00002 --relu_stable 1e-4 --test_freq 50
# verify -- replace the --load_model path with the desired model
python verify/certify.py --dataset cifar10 --net convmedbig_flat_2_2_4_250 --test_eps 0.00784313725 --ib_batch_size 2000 --oval_bab_config ./bab_configs/colt_models.json --oval_bab_timeout 1800 --load_model ./saved_models/cifar_2_255_ibpr.pkl
```

### CIFAR-10 - 8/255
```
# train
python train/train_main.py --train_mode train --dataset cifar10 --net convmed_flat_2_4_250 --train_batch 150 --test_batch 100 --train_eps 0.031372549 --start_eps_factor 1.7 --anneal --train_att_n_steps 8 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.035 --opt sgd --lr 1e-2 --lr_factor 0.95 --cont_lr_decay --cont_lr_mix --mix --mix_epochs 600 --n_epochs 800 --l1_reg 0.00001 --relu_stable 5e-3 --relu_stable_ub_mask --test_freq 50
# verify -- replace the --load_model path with the desired model
python verify/certify.py --dataset cifar10 --net convmed_flat_2_4_250 --test_eps 0.031372549 --ib_batch_size 2000 --oval_bab_config ./bab_configs/colt_models.json --oval_bab_timeout 1800 --load_model ./saved_models/cifar_8_255_ibprm.pkl
```
