#!/usr/bin/env python

import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
import numpy
import numpy as np
from collections import OrderedDict
import cPickle as pickle

from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


rng = np.random.RandomState(1234)
floatX = config.floatX

def shared_to_cpu(shared_params, params):
    for k, v in shared_params.iteritems():
        params[k] = v.get_value()

def cpu_to_shared(params, shared_params):
    for k, v in params.iteritems():
        shared_params[k].set_value(v)

def save_model(filename, options, params, shared_params=None):
    if not shared_params == None:
        shared_to_cpu(shared_params, params);
    model = OrderedDict()
    model['options'] = options
    model['params'] = params
    pickle.dump(model, open(filename, 'w'))

def load_model(filename):
    model = pickle.load(open(filename, 'rb'))
    options = model['options']
    params = model['params']
    shared_params = init_shared_params(params)
    return options, params, shared_params
    # return options, params, shared_params


def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are
    orthogonal.
    """
    W = rng.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')


def init_weight(n, d, options):
    ''' initialize weight matrix
    options['init_type'] determines
    gaussian or uniform initlizaiton
    '''
    if options['init_type'] == 'gaussian':
        return (rng.randn(n, d).astype(floatX)) * options['std']
    elif options['init_type'] == 'uniform':
        # [-range, range]
        return ((rng.rand(n, d) * 2 - 1) * \
                options['range']).astype(floatX)

def init_convweight(w_shape, options):
    ''' initialize weight matrix of convolutional layer
    '''
    if options['init_type'] == 'gaussian':
        return rng.randn(*w_shape).astype(floatX) * options['std']
    elif options['init_type'] == 'uniform':
        return ((rng.rand(*w_shape) * 2 - 1) * options['range']).astype(floatX)

layers = {'ff': ('init_fflayer', 'fflayer'),
          'lstm': ('init_lstm_layer', 'lstm_layer'),
          'lstm_append': (None, 'lstm_append_layer')}

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

def upsample(x, factor):
    """
    Upsamples last two dimensions of symbolic theano tensor.
    x: symbolic theano tensor
        variable to upsample
    factor: int
        upsampling factor
    """
    x_1 = T.extra_ops.repeat(x, factor, axis=x.ndim-2)
    x_2 = T.extra_ops.repeat(x_1, factor, axis=x.ndim-1)
    return x_2

def zero_pad(tensor, output_size):
    zero_padding = T.zeros((tensor.shape[0], tensor.shape[1], output_size[0], output_size[1]))
    pad1 = (output_size[0] - tensor.shape[2])/2
    pad2 = (output_size[1] - tensor.shape[3])/2
    return T.set_subtensor(zero_padding[:, :, pad1:-pad1, pad2:-pad2], tensor)

def gkern(kernlen=14, nsig=[1,1], rot = None):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    res = gaussian_filter(inp, nsig)
    if rot != None:
        res = rotate(res, rot)
    return res

def create_n_gaussian_blobs(width, n):
    res = np.zeros((1, width, width))
    k = int(np.sqrt(n)) + 1
    for i in range(1, k):
        for j in range(1, k):
            res = np.concatenate([res,
                                  gkern(kernlen=width, nsig=[i,j])[None, :, :]],
                                 axis=0)
    return res[1:]

# initialize the parmaters
def init_params(options):
    ''' Initialize all the parameters
    '''
    params = OrderedDict()
    n_words = options['n_words']
    n_emb = options['n_emb']
    n_dim = options['n_dim']
    n_attention = options['n_attention']
    n_image_feat = options['n_image_feat']
    n_common_feat = options['n_common_feat']
    n_output = options['n_output']

    # embedding weights
    # params['w_emb'] = init_weight(n_words, n_emb, options)
    ## use the same initialization as BOW
    params['w_emb_shared'] = ((rng.rand(n_words, n_emb) * 2 - 1) * 0.5).astype(floatX)

    n_filter = 0
    if options['use_unigram_conv']:
        params = init_fflayer(params, n_emb, options['num_filter_unigram'],
                              options, prefix='conv_unigram_shared')
        n_filter += options['num_filter_unigram']
    if options['use_bigram_conv']:
        params = init_fflayer(params, 2 * n_emb, options['num_filter_bigram'],
                              options, prefix='conv_bigram_shared')
        n_filter += options['num_filter_bigram']
    if options['use_trigram_conv']:
        params = init_fflayer(params, 3 * n_emb, options['num_filter_trigram'],
                              options, prefix='conv_trigram_shared')
        n_filter += options['num_filter_trigram']

    params = init_fflayer(params, n_image_feat, n_filter, options,
                          prefix='image_mlp_shared')

    # attention model based parameters
    params = init_fflayer(params, n_filter, n_attention, options,
                          prefix='image_att_mlp_1_shared')
    params = init_fflayer(params, n_filter, n_attention, options,
                          prefix='sent_att_mlp_1_shared')

    if options['maps_first_att_layer']:
        params = init_convlayer(params, (8, n_image_feat, 1, 1), options, prefix='saliency_inception_0_1x1')
        params = init_convlayer(params, (8, n_image_feat, 1, 1), options, prefix='saliency_inception_1_1x1')
        params = init_convlayer(params, (16, 8, 3, 3), options, prefix='saliency_inception_1_3x3')
        params = init_convlayer(params, (2, n_image_feat, 1, 1), options, prefix='saliency_inception_2_1x1')
        params = init_convlayer(params, (4, 2, 3, 3), options, prefix='saliency_inception_2_3x3')
        params = init_convlayer(params, (4, n_image_feat, 1, 1), options, prefix='saliency_inception_3_1x1')

        params = init_convlayer(params, (8, 32, 1, 1), options, prefix='saliency_inception_0_1x1_2')
        params = init_convlayer(params, (8, 32, 1, 1), options, prefix='saliency_inception_1_1x1_2')
        params = init_convlayer(params, (16, 8, 3, 3), options, prefix='saliency_inception_1_3x3_2')
        params = init_convlayer(params, (2, 32, 1, 1), options, prefix='saliency_inception_2_1x1_2')
        params = init_convlayer(params, (4, 2, 3, 3), options, prefix='saliency_inception_2_3x3_2')
        params = init_convlayer(params, (4, 32, 1, 1), options, prefix='saliency_inception_3_1x1_2')

        if options['use_LB']:
            params = init_LBconvlayer(params, (32, 32, 5, 5), 16, 14, options, prefix='saliency_LB_conv')
            params = init_LBconvlayer(params, (32, 32, 5, 5), 16, 12, options, prefix='saliency_LB_conv_2')
        else:
            params = init_convlayer(params, (32, 32, 5, 5), options, prefix='saliency_conv')
            # params = init_convlayer(params, (32, 32, 5, 5), options, prefix='conv_2')

        params = init_convlayer(params, (1, 32, 1, 1), options, prefix='saliency_combined_att_mlp_1')

        params = init_fflayer(params, 196, 196, options, prefix='answer_gate')

    params = init_fflayer(params, n_attention, 1, options,
                          prefix='combined_att_mlp_1_shared_second')
    params = init_fflayer(params, n_filter, n_attention, options,
                          prefix='image_att_mlp_2_shared_second')
    params = init_fflayer(params, n_filter, n_attention, options,
                          prefix='sent_att_mlp_2_shared_second')

    if options['maps_second_att_layer']:
        params = init_convlayer(params, (8, n_image_feat, 1, 1), options, prefix='saliency_inception_0_1x1')
        params = init_convlayer(params, (8, n_image_feat, 1, 1), options, prefix='saliency_inception_1_1x1')
        params = init_convlayer(params, (16, 8, 3, 3), options, prefix='saliency_inception_1_3x3')
        params = init_convlayer(params, (2, n_image_feat, 1, 1), options, prefix='saliency_inception_2_1x1')
        params = init_convlayer(params, (4, 2, 3, 3), options, prefix='saliency_inception_2_3x3')
        params = init_convlayer(params, (4, n_image_feat, 1, 1), options, prefix='saliency_inception_3_1x1')

        params = init_convlayer(params, (8, 32, 1, 1), options, prefix='saliency_inception_0_1x1_2')
        params = init_convlayer(params, (8, 32, 1, 1), options, prefix='saliency_inception_1_1x1_2')
        params = init_convlayer(params, (16, 8, 3, 3), options, prefix='saliency_inception_1_3x3_2')
        params = init_convlayer(params, (2, 32, 1, 1), options, prefix='saliency_inception_2_1x1_2')
        params = init_convlayer(params, (4, 2, 3, 3), options, prefix='saliency_inception_2_3x3_2')
        params = init_convlayer(params, (4, 32, 1, 1), options, prefix='saliency_inception_3_1x1_2')

        if options['use_LB']:
            params = init_LBconvlayer(params, (32, 32, 5, 5), 16, 14, options, prefix='saliency_LB_conv')
            params = init_LBconvlayer(params, (32, 32, 5, 5), 16, 12, options, prefix='saliency_LB_conv_2')
        else:
            params = init_convlayer(params, (32, 32, 3, 3), options, prefix='saliency_conv')
            params = init_convlayer(params, (32, 32, 3, 3), options, prefix='saliency_conv_2')

        params = init_convlayer(params, (1, 32, 1, 1), options, prefix='saliency_combined_att_mlp_2')

    params = init_fflayer(params, n_attention, 1, options,
                          prefix='answer_combined_att_mlp_2_pos')
    params = init_fflayer(params, n_attention, 1, options,
                          prefix='answer_combined_att_mlp_2_neg')

    # if options['maps_second_att_layer']:
    #     params = init_fflayer(params, 196*2, 196, options,
    #                           prefix='answer_combined_att_mlp_2_2')

    for i in range(options['combined_num_mlp']):
        if i == 0 and options['combined_num_mlp'] == 1:
            params = init_fflayer(params, n_filter, n_output,
                                  options, prefix='answer_combined_mlp_%d'%(i))
        elif i == 0 and options['combined_num_mlp'] != 1:
            params = init_fflayer(params, n_filter, n_common_feat,
                                  options, prefix='answer_combined_mlp_%d'%(i))
        elif i == options['combined_num_mlp'] - 1 :
            params = init_fflayer(params, n_common_feat, n_output,
                                  options, prefix='answer_combined_mlp_%d'%(i))
        else:
            params = init_fflayer(params, n_common_feat, n_common_feat,
                                  options, prefix='answer_combined_mlp_%d'%(i))

    return params

def init_shared_params(params):
    ''' return a shared version of all parameters
    '''
    shared_params = OrderedDict()
    for k, p in params.iteritems():
        shared_params[k] = theano.shared(params[k], name = k)

    return shared_params

def init_shared_params_maps(shared_params, options):
    ''' return a shared version of shared task parameters
    '''
    shared_params_maps = OrderedDict()

    for k, p in shared_params.iteritems():
        if 'shared' in k or 'saliency' in k:
            if options['maps_first_att_layer'] and 'second' in k:
                pass
            else:
                shared_params_maps[k] = shared_params[k]

    return shared_params_maps

def init_shared_params_answer(shared_params, options):
    ''' return a shared version of shared task parameters
    '''
    shared_params_answer = OrderedDict()

    for k, p in shared_params.iteritems():
        if 'shared' in k or 'answer' in k:
            shared_params_answer[k] = shared_params[k]

    return shared_params_answer

# activation function for ff layer
def sigmoid(x):
    return T.nnet.sigmoid(x)

def tanh(x):
    return T.tanh(x)

def relu(x):
    return T.maximum(x, np.float32(0.))

def linear(x):
    return x

def init_fflayer(params, nin, nout, options, prefix='ff'):
    ''' initialize ff layer
    '''
    params[prefix + '_w'] = init_weight(nin, nout, options)
    params[prefix + '_b'] = np.zeros(nout, dtype='float32')
    return params

def fflayer(shared_params, x, options, prefix='ff', act_func='tanh'):
    ''' fflayer: multiply weight then add bias
    '''
    return eval(act_func)(T.dot(x, shared_params[prefix + '_w']) +
                          shared_params[prefix + '_b'])

def init_convlayer(params, w_shape, options, prefix='conv'):
    ''' init conv layer
    '''
    params[prefix + '_w'] = init_convweight(w_shape, options)
    params[prefix + '_b'] = np.zeros(w_shape[0]).astype(floatX)
    return params

def convlayer(shared_params, x, options, prefix='conv', act_func='relu', pad=0, size_holes=1):
    return eval(act_func)(conv2d(x, shared_params[prefix + '_w'], border_mode=(pad, pad), filter_dilation=(size_holes, size_holes)) +
                          shared_params[prefix + '_b'].dimshuffle('x', 0, 'x', 'x'))

def init_LBconvlayer(params, w_shape, n_blobs, width, options, prefix='conv'):
    ''' init Location Biased conv layer
    '''
    params[prefix + '_w'] = init_convweight(w_shape, options)
    params[prefix + '_w*'] = init_convweight((w_shape[0], n_blobs, w_shape[2], w_shape[3]), options)
    params[prefix + '_L'] = create_n_gaussian_blobs(width, n_blobs).astype('float32')
    params[prefix + '_b'] = np.zeros(w_shape[0]).astype(floatX)
    return params

def LBconvlayer(shared_params, params, x, options, prefix='conv', act_func='relu', pad=0, size_holes=1):
    L = params[prefix + '_L'][np.newaxis, :, :, :]
    return eval(act_func)(conv2d(x, shared_params[prefix + '_w'], border_mode=(pad, pad), filter_dilation=(size_holes, size_holes)) +
                          conv2d(L, shared_params[prefix + '_w*'], border_mode=(pad, pad), filter_dilation=(size_holes, size_holes)) +
                          shared_params[prefix + '_b'].dimshuffle('x', 0, 'x', 'x'))

def maxpool_layer(shared_params, x, maxpool_shape, options, s=1, padding=0):
    return pool.pool_2d(x, maxpool_shape, ignore_border=True, stride=(s,s), pad=(padding, padding))

def dropout_layer(x, dropout, trng, drop_ratio=0.5):
    ''' dropout layer
    '''
    x_drop = T.switch(dropout,
                      (x * trng.binomial(x.shape,
                                         p = 1 - drop_ratio,
                                         n = 1,
                                         dtype = x.dtype) \
                       / (numpy.float32(1.0) - drop_ratio)),
                      x)
    return x_drop

def similarity_layer(feat, feat_seq):
    def _step(x, y):
        return T.sum(x*y, axis=1) / (T.sqrt(T.sum(x*x, axis=1) * \
                                            T.sum(y*y, axis=1))
                                     + np.float(1e-7))
    similarity, updates = theano.scan(fn = _step,
                                      sequences = [feat_seq],
                                      outputs_info = None,
                                      non_sequences = [feat],
                                      n_steps = feat_seq.shape[0])
    return similarity


def build_model(shared_params, params, options):
    trng = RandomStreams(1234)
    drop_ratio = options['drop_ratio']
    batch_size = options['batch_size']
    n_dim = options['n_dim']

    w_emb = shared_params['w_emb_shared']

    dropout = theano.shared(numpy.float32(0.))
    image_feat = T.ftensor3('image_feat')
    # batch_size x T
    input_idx = T.imatrix('input_idx')
    input_mask = T.matrix('input_mask')
    # label is the TRUE label

    map_label = T.matrix('map_label')
    label = T.ivector('label')
    empty_word = theano.shared(value=np.zeros((1, options['n_emb']),
                                              dtype='float32'),
                               name='empty_word')
    w_emb_extend = T.concatenate([empty_word, shared_params['w_emb_shared']],
                                 axis=0)
    input_emb = w_emb_extend[input_idx]

    # a trick here, set the maxpool_h/w to be large
    # maxpool_shape = (options['maxpool_h'], options['maxpool_w'])

    # turn those appending words into zeros
    # batch_size x T x n_emb
    input_emb = input_emb * input_mask[:, :, None]
    if options['sent_drop']:
        input_emb = dropout_layer(input_emb, dropout, trng, drop_ratio)

    if options['use_unigram_conv']:
        unigram_conv_feat = fflayer(shared_params, input_emb, options,
                                    prefix='conv_unigram_shared',
                                    act_func=options.get('sent_conv_act', 'tanh'))
        unigram_pool_feat = unigram_conv_feat.max(axis=1)
    if options['use_bigram_conv']:
        idx = T.concatenate([T.arange(input_emb.shape[1])[:-1],
                             T.arange(input_emb.shape[1])[1:]]).reshape((2, input_emb.shape[1] - 1)).transpose().flatten()
        bigram_emb = T.reshape(input_emb[:, idx, :], (input_emb.shape[0],
                                                      input_emb.shape[1] - 1,
                                                      2 * input_emb.shape[2]))
        bigram_conv_feat = fflayer(shared_params, bigram_emb,
                                   options, prefix='conv_bigram_shared',
                                   act_func=options.get('sent_conv_act', 'tanh'))
        bigram_pool_feat = bigram_conv_feat.max(axis=1)
    if options['use_trigram_conv']:
        idx = T.concatenate([T.arange(input_emb.shape[1])[:-2],
                             T.arange(input_emb.shape[1])[1:-1],
                             T.arange(input_emb.shape[1])[2:]]).reshape((3, input_emb.shape[1] - 2)).transpose().flatten()
        trigram_emb = T.reshape(input_emb[:, idx, :], (input_emb.shape[0],
                                                      input_emb.shape[1] - 2,
                                                      3 * input_emb.shape[2]))
        trigram_conv_feat = fflayer(shared_params, trigram_emb,
                                    options, prefix='conv_trigram_shared',
                                    act_func=options.get('sent_conv_act', 'tanh'))
        trigram_pool_feat = trigram_conv_feat.max(axis=1)  #

    pool_feat = T.concatenate([unigram_pool_feat,
                               bigram_pool_feat,
                               trigram_pool_feat], axis=1)

    image_feat_down = fflayer(shared_params, image_feat, options,
                              prefix='image_mlp_shared',
                              act_func=options.get('image_mlp_act',
                                                   'tanh'))
    if options.get('use_before_attention_drop', False):
        image_feat_down = dropout_layer(image_feat_down, dropout, trng, drop_ratio)
        pool_feat = dropout_layer(pool_feat, dropout, trng, drop_ratio)

    # attention model begins here
    # first layer attention model
    image_feat_attention_1 = fflayer(shared_params, image_feat_down, options,
                                     prefix='image_att_mlp_1_shared',
                                     act_func=options.get('image_att_mlp_act',
                                                          'tanh'))
    pool_feat_attention_1 = fflayer(shared_params, pool_feat, options,
                                    prefix='sent_att_mlp_1_shared',
                                    act_func=options.get('sent_att_mlp_act',
                                                         'tanh'))
    combined_feat_attention_1 = image_feat_attention_1 + \
                                pool_feat_attention_1[:, None, :]
    if options['use_attention_drop']:
        combined_feat_attention_1 = dropout_layer(combined_feat_attention_1,
                                                  dropout, trng, drop_ratio)

    if options['maps_first_att_layer']:
        combine_reshaped = combined_feat_attention_1.swapaxes(1,2).reshape((combined_feat_attention_1.shape[0],
                                                                            combined_feat_attention_1.shape[2],
                                                                            14,
                                                                            14))
        # INCEPTION LAYER
        saliency_inception_0_1x1 = convlayer(shared_params,
                                             combine_reshaped,
                                             options,
                                             prefix='saliency_inception_0_1x1')

        saliency_inception_1_1x1 = convlayer(shared_params,
                                             combine_reshaped,
                                             options,
                                             prefix='saliency_inception_1_1x1')
        saliency_inception_1_3x3 = convlayer(shared_params,
                                             saliency_inception_1_1x1,
                                             options,
                                             prefix='saliency_inception_1_3x3',
                                             pad=1)

        saliency_inception_2_1x1 = convlayer(shared_params,
                                             combine_reshaped,
                                             options,
                                             prefix='saliency_inception_2_1x1')
        saliency_inception_2_3x3 = convlayer(shared_params,
                                             saliency_inception_2_1x1,
                                             options,
                                             prefix='saliency_inception_2_3x3',
                                             pad=2,
                                             size_holes=2)

        saliency_inception_3_maxpool = maxpool_layer(shared_params,
                                                     combine_reshaped,
                                                     (3,3),
                                                     options,
                                                     padding=1)
        saliency_inception_3_1x1 = convlayer(shared_params,
                                             saliency_inception_3_maxpool,
                                             options,
                                             prefix='saliency_inception_3_1x1')

        saliency_inception = T.concatenate([saliency_inception_0_1x1,
                                            saliency_inception_1_3x3,
                                            saliency_inception_2_3x3,
                                            saliency_inception_3_1x1], axis=1)

        saliency_inception_0_1x1 = convlayer(shared_params,
                                             saliency_inception,
                                             options,
                                             prefix='saliency_inception_0_1x1_2')

        saliency_inception_1_1x1 = convlayer(shared_params,
                                             saliency_inception,
                                             options,
                                             prefix='saliency_inception_1_1x1_2')
        saliency_inception_1_3x3 = convlayer(shared_params,
                                             saliency_inception_1_1x1,
                                             options,
                                             prefix='saliency_inception_1_3x3_2',
                                             pad=1)

        saliency_inception_2_1x1 = convlayer(shared_params,
                                             saliency_inception,
                                             options,
                                             prefix='saliency_inception_2_1x1_2')
        saliency_inception_2_3x3 = convlayer(shared_params,
                                             saliency_inception_2_1x1,
                                             options,
                                             prefix='saliency_inception_2_3x3_2',
                                             pad=2,
                                             size_holes=2)

        saliency_inception_3_maxpool = maxpool_layer(shared_params,
                                                     saliency_inception,
                                                     (3,3),
                                                     options,
                                                     padding=1)
        saliency_inception_3_1x1 = convlayer(shared_params,
                                             saliency_inception_3_maxpool,
                                             options,
                                             prefix='saliency_inception_3_1x1_2')

        saliency_inception = T.concatenate([saliency_inception_0_1x1,
                                            saliency_inception_1_3x3,
                                            saliency_inception_2_3x3,
                                            saliency_inception_3_1x1], axis=1)

        if options['use_LB']:
            saliency_LBconv = LBconvlayer(shared_params,
                                          params,
                                          saliency_inception,
                                          options,
                                          prefix='saliency_LB_conv',
                                          pad=8,
                                          size_holes=6)

            saliency_feat = T.nnet.abstract_conv.bilinear_upsampling(saliency_LBconv, 2)

            # saliency_LBconv = LBconvlayer(shared_params,
            #                               params,
            #                               saliency_feat,
            #                               options,
            #                               prefix='saliency_LB_conv_2',
            #                               pad=8,
            #                               size_holes=6)
            #
            # saliency_feat = T.nnet.abstract_conv.bilinear_upsampling(saliency_LBconv, 3)
        else:
            saliency_conv = convlayer(shared_params,
                                      saliency_inception,
                                      options,
                                      prefix='saliency_conv',
                                      pad=8,
                                      size_holes=6)

            saliency_feat = T.nnet.abstract_conv.bilinear_upsampling(saliency_conv, 2)

            saliency_conv = convlayer(shared_params,
                                          params,
                                          saliency_feat,
                                          options,
                                          prefix='saliency_conv_2',
                                          pad=8,
                                          size_holes=6)

            saliency_feat = T.nnet.abstract_conv.bilinear_upsampling(saliency_conv, 3)

        saliency_feat = dropout_layer(saliency_feat,
                                      dropout, trng, drop_ratio)

        saliency_feat = convlayer(shared_params,
                                  saliency_feat,
                                  options,
                                  prefix='saliency_combined_att_mlp_1',
                                  act_func=options.get('combined_att_mlp_act', 'tanh'),
                                  pad=1)

        saliency_feat = saliency_feat.reshape((saliency_feat.shape[0],
                                               saliency_feat.shape[1],
                                               saliency_feat.shape[2] * saliency_feat.shape[3]))
        saliency_feat = saliency_feat.swapaxes(1,2)

        saliency_attention = T.nnet.softmax(saliency_feat[:, :, 0])

        saliency_attention_section = saliency_attention[:map_label.shape[0]]

        if options['use_kl']:
            if options['reverse_kl']:
                prob_map = T.sum(T.log(saliency_attention_section / map_label)*saliency_attention_section, axis=1)
            else:
                prob_map = T.sum(T.log(map_label / saliency_attention_section)*map_label, axis=1)
        else:
            prob_map = -T.sum(T.log(saliency_attention_section)*map_label, axis=1)

        map_cost = T.mean(prob_map)

        answer_gate = fflayer(shared_params,
                              saliency_feat[:, :, 0], options,
                              prefix='answer_gate',
                              act_func='sigmoid')

        combined_feat_attention_1 = fflayer(shared_params,
                                            combined_feat_attention_1, options,
                                            prefix='combined_att_mlp_1_shared_second',
                                            act_func=options.get(
                                                'combined_att_mlp_act', 'tanh'))

        combined_feat_attention_1 = answer_gate[:, :, None]*saliency_feat + combined_feat_attention_1
        prob_attention_1 = T.nnet.softmax(combined_feat_attention_1[:, :, 0])
        image_feat_ave_1 = (prob_attention_1[:, :, None] * image_feat_down).sum(axis=1)

    else:

        combined_feat_attention_1 = fflayer(shared_params,
                                            combined_feat_attention_1, options,
                                            prefix='combined_att_mlp_1_shared_second',
                                            act_func=options.get(
                                                'combined_att_mlp_act', 'tanh'))
        prob_attention_1 = T.nnet.softmax(combined_feat_attention_1[:, :, 0])
        image_feat_ave_1 = (prob_attention_1[:, :, None] * image_feat_down).sum(axis=1)

    combined_hidden_1 = image_feat_ave_1 + pool_feat
    # second layer attention model

    image_feat_attention_2 = fflayer(shared_params, image_feat_down, options,
                                     prefix='image_att_mlp_2_shared_second',
                                     act_func=options.get('image_att_mlp_act',
                                                          'tanh'))
    pool_feat_attention_2 = fflayer(shared_params, combined_hidden_1, options,
                                    prefix='sent_att_mlp_2_shared_second',
                                    act_func=options.get('sent_att_mlp_act',
                                                         'tanh'))
    combined_feat_attention_2 = image_feat_attention_2 + \
                                pool_feat_attention_2[:, None, :]

    if options['use_attention_drop']:
        combined_feat_attention_2 = dropout_layer(combined_feat_attention_2,
                                                  dropout, trng, drop_ratio)

    if options['maps_second_att_layer']:
        combine_reshaped = combined_feat_attention_2.swapaxes(1,2).reshape((combined_feat_attention_2.shape[0],
                                                                            combined_feat_attention_2.shape[2],
                                                                            14,
                                                                            14))
        # INCEPTION LAYER
        saliency_inception_0_1x1 = convlayer(shared_params,
                                             combine_reshaped,
                                             options,
                                             prefix='saliency_inception_0_1x1')

        saliency_inception_1_1x1 = convlayer(shared_params,
                                             combine_reshaped,
                                             options,
                                             prefix='saliency_inception_1_1x1')
        saliency_inception_1_3x3 = convlayer(shared_params,
                                             saliency_inception_1_1x1,
                                             options,
                                             prefix='saliency_inception_1_3x3',
                                             pad=1)

        saliency_inception_2_1x1 = convlayer(shared_params,
                                             combine_reshaped,
                                             options,
                                             prefix='saliency_inception_2_1x1')
        saliency_inception_2_3x3 = convlayer(shared_params,
                                             saliency_inception_2_1x1,
                                             options,
                                             prefix='saliency_inception_2_3x3',
                                             pad=2,
                                             size_holes=2)

        saliency_inception_3_maxpool = maxpool_layer(shared_params,
                                                     combine_reshaped,
                                                     (3,3),
                                                     options,
                                                     padding=1)
        saliency_inception_3_1x1 = convlayer(shared_params,
                                             saliency_inception_3_maxpool,
                                             options,
                                             prefix='saliency_inception_3_1x1')

        saliency_inception = T.concatenate([saliency_inception_0_1x1,
                                            saliency_inception_1_3x3,
                                            saliency_inception_2_3x3,
                                            saliency_inception_3_1x1], axis=1)

        saliency_inception_0_1x1 = convlayer(shared_params,
                                             saliency_inception,
                                             options,
                                             prefix='saliency_inception_0_1x1_2')

        saliency_inception_1_1x1 = convlayer(shared_params,
                                             saliency_inception,
                                             options,
                                             prefix='saliency_inception_1_1x1_2')
        saliency_inception_1_3x3 = convlayer(shared_params,
                                             saliency_inception_1_1x1,
                                             options,
                                             prefix='saliency_inception_1_3x3_2',
                                             pad=1)

        saliency_inception_2_1x1 = convlayer(shared_params,
                                             saliency_inception,
                                             options,
                                             prefix='saliency_inception_2_1x1_2')
        saliency_inception_2_3x3 = convlayer(shared_params,
                                             saliency_inception_2_1x1,
                                             options,
                                             prefix='saliency_inception_2_3x3_2',
                                             pad=2,
                                             size_holes=2)

        saliency_inception_3_maxpool = maxpool_layer(shared_params,
                                                     saliency_inception,
                                                     (3,3),
                                                     options,
                                                     padding=1)
        saliency_inception_3_1x1 = convlayer(shared_params,
                                             saliency_inception_3_maxpool,
                                             options,
                                             prefix='saliency_inception_3_1x1_2')

        saliency_inception = T.concatenate([saliency_inception_0_1x1,
                                            saliency_inception_1_3x3,
                                            saliency_inception_2_3x3,
                                            saliency_inception_3_1x1], axis=1)

        if options['use_LB']:
            saliency_LBconv = LBconvlayer(shared_params,
                                          params,
                                          saliency_inception,
                                          options,
                                          prefix='saliency_LB_conv',
                                          pad=8,
                                          size_holes=6)

            saliency_feat = T.nnet.abstract_conv.bilinear_upsampling(saliency_LBconv, 2)

            saliency_LBconv = LBconvlayer(shared_params,
                                          params,
                                          saliency_feat,
                                          options,
                                          prefix='saliency_LB_conv_2',
                                          pad=8,
                                          size_holes=6)

            saliency_feat = T.nnet.abstract_conv.bilinear_upsampling(saliency_LBconv, 3)
        else:
            saliency_feat = convlayer(shared_params,
                                      saliency_inception,
                                      options,
                                      prefix='saliency_conv',
                                      pad=3,
                                      size_holes=3)

            saliency_feat = convlayer(shared_params,
                                      saliency_feat,
                                      options,
                                      prefix='saliency_conv_2',
                                      pad=3,
                                      size_holes=3)

        saliency_feat = dropout_layer(saliency_feat,
                                      dropout, trng, drop_ratio)

        saliency_feat = convlayer(shared_params,
                                  saliency_feat,
                                  options,
                                  prefix='saliency_combined_att_mlp_2',
                                  act_func=options.get('combined_att_mlp_act', 'tanh'))

        saliency_feat = saliency_feat.reshape((saliency_feat.shape[0],
                                               saliency_feat.shape[1],
                                               saliency_feat.shape[2] * saliency_feat.shape[3]))
        saliency_feat = saliency_feat.swapaxes(1,2)

        saliency_attention = T.nnet.softmax(saliency_feat[:, :, 0])

        saliency_attention_section = saliency_attention[:map_label.shape[0]]

        if options['use_kl']:
            if options['reverse_kl']:
                prob_map = T.sum(T.log(saliency_attention_section / map_label)*saliency_attention_section, axis=1)
            else:
                prob_map = T.sum(T.log(map_label / saliency_attention_section)*map_label, axis=1)
        else:
            prob_map = -T.sum(T.log(saliency_attention_section)*map_label, axis=1)

        map_cost = T.mean(prob_map)

        combined_feat_attention_2_pos = fflayer(shared_params,
                                                combined_feat_attention_2, options,
                                                prefix='answer_combined_att_mlp_2_pos',
                                                act_func=options.get(
                                                    'combined_att_mlp_act', 'tanh'))

        combined_feat_attention_2_neg = fflayer(shared_params,
                                                combined_feat_attention_2, options,
                                                prefix='answer_combined_att_mlp_2_neg',
                                                act_func=options.get(
                                                    'combined_att_mlp_act', 'tanh'))

        combined_feat_attention_2 = saliency_attention * combined_feat_attention_2_pos + (1-saliency_attention) * combined_feat_attention_2_neg

        prob_attention_2 = T.nnet.softmax(combined_feat_attention_2[:, :, 0])
        image_feat_ave_2 = (prob_attention_2[:, :, None] * image_feat_down).sum(axis=1)

    else:
        combined_feat_attention_2 = fflayer(shared_params,
                                            combined_feat_attention_2, options,
                                            prefix='answer_combined_att_mlp_2',
                                            act_func=options.get(
                                                'combined_att_mlp_act', 'tanh'))
        prob_attention_2 = T.nnet.softmax(combined_feat_attention_2[:, :, 0])
        image_feat_ave_2 = (prob_attention_2[:, :, None] * image_feat_down).sum(axis=1)

    if options.get('use_final_image_feat_only', False):
        combined_hidden = image_feat_ave_2 + pool_feat
    else:
        combined_hidden = image_feat_ave_2 + combined_hidden_1

    for i in range(options['combined_num_mlp']):
        if options.get('combined_mlp_drop_%d'%(i), False):
            combined_hidden = dropout_layer(combined_hidden, dropout, trng,
                                            drop_ratio)
        if i == options['combined_num_mlp'] - 1:
            combined_hidden = fflayer(shared_params, combined_hidden, options,
                                      prefix='answer_combined_mlp_%d'%(i),
                                      act_func='linear')
        else:
            combined_hidden = fflayer(shared_params, combined_hidden, options,
                                      prefix='answer_combined_mlp_%d'%(i),
                                      act_func=options.get('combined_mlp_act_%d'%(i),
                                                           'tanh'))

    # drop the image output
    prob = T.nnet.softmax(combined_hidden)
    prob_y = prob[T.arange(prob.shape[0]), label]
    pred_label = T.argmax(prob, axis=1)
    # sum or mean?
    cost = -T.mean(T.log(prob_y))
    accu = T.mean(T.eq(pred_label, label))

    # return image_feat, input_idx, input_mask, \
        # label, dropout, cost, accu
    return image_feat, input_idx, input_mask, \
        label, dropout, cost, accu, pred_label, \
        prob_attention_1, prob_attention_2, map_cost, map_label, saliency_attention

    # return image_feat, input_idx, input_mask, \
        # label, dropout, cost, accu, pred_label, \
        # image_feat_down, pool_feat

    # for debug
    # return image_feat, input_idx, input_mask, label, dropout, cost, accu, \
        # input_emb, bigram_emb, trigram_emb, trigram_pool_feat, pool_feat
