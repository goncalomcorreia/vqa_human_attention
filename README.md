# vqa-human-attention
Adapted from
[Stacked attention networks for image question answering](http://arxiv.org/abs/1511.02274).

## Dependencies
The code is in python and uses Theano package.
- Python 2.7
- Theano
- Numpy
- h5py


## Usage

Data can be found in project space /afs/inf.ed.ac.uk/group/synproc/Goncalo/ in
in folders 'data_vqa' and 'data_att_maps'

To train a model,
```
cd src/scripts; python mtl_san_deepfix.py
```

Scripts in folder src/scripts have the same name as the respective model described in the dissertation.
