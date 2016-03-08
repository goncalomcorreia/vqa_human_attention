# imageqa-san
Source code for
[Stacked attention networks for image question answering](http://arxiv.org/abs/1511.02274).

Joint collaboration between CMU and MSR.

## Dependencies
The code is in python and uses Theano package.
- Python 2.7
- Theano
- Numpy
- h5py


## Usage
```
cd src; python san_att_conv_twolayer.py
```
to start training.

## Reference
If you use this code as part of your research, please cite our paper
**'Stacked Attention Netowrks for Image Question Answering'**,
Zichao Yang, Xiaodong He, Jianfeng Gao, Li Deng and Alex Smola.
To appear in CVPR 2016.

```
@article{YangHGDS15,
author    = {Zichao Yang and
Xiaodong He and
Jianfeng Gao and
Li Deng and
Alexander J. Smola},
title     = {Stacked Attention Networks for Image Question Answering},
journal   = {CoRR},
volume    = {abs/1511.02274},
year      = {2015},
url       = {http://arxiv.org/abs/1511.02274},
}
```
