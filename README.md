# WGAN-TS
PyTorch implementation of [A Two-Step Computation of the Exact GAN Wasserstein Distance](http://proceedings.mlr.press/v80/liu18d.html), ICML 2018.
# Usage
DATASET: mnist/cifar10/lsun <br>
DATASET_PATH: path to the root folder of a dataset <br>
1. Install dependency: [CVXOPT](https://cvxopt.org/) for linear programming. <br> 
2. Download your dataset, unzip it and put it in DATASET_PATH. <br>
3. run code using 
```
python WGAN_TS.py --dataset DATASET --dataroot DATASET_PATH --BN_G
```
You are welcome to cite our work using:
```
@inproceedings{liu2018two,
  title={A Two-Step Computation of the Exact GAN Wasserstein Distance},
  author={Liu, Huidong and Xianfeng, GU and Samaras, Dimitris},
  booktitle={International Conference on Machine Learning},
  pages={3165--3174},
  year={2018}
}
```
