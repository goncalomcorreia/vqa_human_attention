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

## Files
```
src/
    san_att_conv_twolayer.py: main file to train two layer san with conv
    san_att_conv_twolayer_theano.py: model building for two layer san with conv
    san_att_lstm_twolayer.py: main file to train two layer san with lstm
    san_att_lstm_twolayer_theano.py: model building for two layer san with lstm
    log.py: log config file
    optimization_weight.py: optimization algorithms
    data_provision_att_vqa.py: data iterator
    data_processing_vqa.py: batch processing before training
```

## Usage
```
python san_att_conv_twolayer.py
```
to start training.

## Reference
If you use this code as part of your research, please cite our paper
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
