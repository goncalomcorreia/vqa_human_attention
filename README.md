# Making Machines More Human: A Multitask Learning Approach to VQA and Human Attention Prediction

My MSc Dissertation Project had the goal of developing a Deep Learning algorithm capable of improving VQA performance of a state-of-the-art architecture while mimicking human attention, using the [VQA-HAT dataset](https://computing.ece.vt.edu/~abhshkdz/vqa-hat/). The Project was successful and some results can be found below.

The Dissertation PDF can be found in this repository - msc-dissertation.pdf.

Code adapted from
[Stacked attention networks for image question answering](http://arxiv.org/abs/1511.02274).

## Dependencies
The code is in python and uses Theano package.
- Python 2.7
- Theano
- Numpy
- h5py


## Usage

To train a model,
```
cd src/scripts; python mtl_san_deepfix.py
```

There is another README.md inside src describing the files there.

## Results

Some results can be found below. "Human Attention and Answer" is the ground-truth. "SAN" is our main baseline - the Stacked Attention Network. Our main algorithm is "MTL SAN+DeepFix", able to improve VQA accuracy of our baseline SAN, while mimicking human attention. Remaining models are different baselines. Thorough explanations can be found in msc-dissertation.pdf

![alt tag](http://i.imgur.com/wO82ecj.jpg)

![alt tag](http://i.imgur.com/Wetfozn.jpg)
