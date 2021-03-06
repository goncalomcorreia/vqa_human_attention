#!/usr/bin/env python


import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')
import datetime
import os
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32,exception_verbosity=high"
import sys
import logging
import argparse
import math
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/')
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/data-providers/')
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/models/')
import log
import numpy as np
np.random.seed(1234)
from optimization_weight import *
from deepfix_att_theano import *
from data_provision_att_vqa_with_maps import *
from data_processing_vqa import *

##################
# initialization #
##################
options = OrderedDict()
# data related
options['data_path'] = '/afs/inf.ed.ac.uk/group/synproc/Goncalo/data_vqa'
options['map_data_path'] = '/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/data_att_maps'
options['feature_file'] = 'trainval_feat.h5'
options['expt_folder'] = '/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/expt/tuning'
options['checkpoint_folder'] = os.path.join(options['expt_folder'], 'checkpoints')
options['model_name'] = 'correct_deepfix'
options['train_split'] = 'trainval1'
options['val_split'] = 'val2'
options['train_split_maps'] = 'train'
options['val_split_maps'] = 'val1'
options['shuffle'] = True
options['reverse'] = False
options['sample_answer'] = True

options['num_region'] = 196
options['region_dim'] = 512

options['n_words'] = 13746
options['n_output'] = 1000

# structure options
options['combined_num_mlp'] = 1
options['combined_mlp_drop_0'] = True
options['combined_mlp_act_0'] = 'linear'
options['sent_drop'] = False
options['use_tanh'] = False
options['use_unigram_conv'] = True
options['use_bigram_conv'] = True
options['use_trigram_conv'] = True

options['use_attention_drop'] = False
options['use_before_attention_drop'] = False

options['use_kl'] = True
options['reverse_kl'] = False
options['task_p'] = 0.8
options['maps_second_att_layer'] = True
options['use_third_att_layer'] = False
options['alt_training'] = True
options['hat_frac'] = 0.2
options['lambda'] = 1
options['use_LB'] = False

# dimensions
options['n_emb'] = 500
options['n_dim'] = 500
options['n_image_feat'] = options['region_dim']
options['n_common_feat'] = 500
options['num_filter_unigram'] = 256
options['num_filter_bigram'] = 512
options['num_filter_trigram'] = 512
options['n_attention'] = 512

# initialization
options['init_type'] = 'uniform'
options['range'] = 0.01
options['std'] = 0.01
options['init_lstm_svd'] = False

# learning parameters
options['optimization'] = 'sgd' # choices
options['batch_size'] = 100
options['lr'] = numpy.float32(1e-1)
options['w_emb_lr'] = numpy.float32(80)
options['momentum'] = numpy.float32(0.9)
options['gamma'] = 1
options['step'] = 10
options['step_start'] = 100
options['max_epochs'] = 20
options['weight_decay'] = 5e-4
options['weight_decay_sub'] = 5e-4
options['decay_rate'] = numpy.float32(0.999)
options['drop_ratio'] = numpy.float32(0.5)
options['smooth'] = numpy.float32(1e-8)
options['grad_clip'] = numpy.float32(0.1)

# log params
options['disp_interval'] = 10
options['eval_interval'] = 100
options['save_interval'] = 500

def get_lr(options, curr_epoch):
    if options['optimization'] == 'sgd':
        power = max((curr_epoch - options['step_start']) / options['step'], 0)
        power = math.ceil(power)
        return options['lr'] * (options['gamma'] ** power)
    else:
        return options['lr']

def train(options):

    logger = logging.getLogger('root')
    logger.info(options)
    logger.info('start training')

    data_provision_att_vqa_maps = DataProvisionAttVqaWithMaps(options['data_path'],
                                                              options['feature_file'],
                                                              options['map_data_path'])

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']

    ###############
    # build model #
    ###############

    params = init_params(options)
    shared_params_maps = init_shared_params(params)

    logger.info(shared_params_maps)

    image_feat, input_idx, input_mask, \
        label, dropout, \
        prob_attention_1, map_cost, map_label = build_model(shared_params_maps, params, options)

    logger.info('finished building model')

    ####################
    # add weight decay #
    ####################
    weight_decay_sub = theano.shared(numpy.float32(options['weight_decay_sub']),\
                                 name = 'weight_decay_sub')

    reg_map = 0

    for k in shared_params_maps.iterkeys():
        if k != 'w_emb':
            reg_map += (shared_params_maps[k]**2).sum()

    reg_map *= weight_decay_sub

    map_reg_cost = map_cost + reg_map

    ###############
    # # gradients #
    ###############
    map_grads = T.grad(map_reg_cost, wrt = shared_params_maps.values())

    grad_buf_maps = [theano.shared(p.get_value() * 0, name='%s_grad_buf' % k )
                     for k, p in shared_params_maps.iteritems()]

    # accumulate the gradients within one batch
    maps_update_grad = [(g_b, g) for g_b, g in zip(grad_buf_maps, map_grads)]
    # need to declare a share variable ??
    grad_clip = options['grad_clip']

    grad_norm_maps = [T.sqrt(T.sum(g_b**2)) for g_b in grad_buf_maps]
    update_clip_maps = [(g_b, T.switch(T.gt(g_norm, grad_clip),
                                  g_b*grad_clip/g_norm, g_b))
                   for (g_norm, g_b) in zip(grad_norm_maps, grad_buf_maps)]

    f_grad_clip_subtask = theano.function(inputs = [],
                                  updates = update_clip_maps)
    f_output_grad_norm = theano.function(inputs = [],
                                         outputs = grad_norm_maps)
    #
    # f_debug = theano.function(inputs = [image_feat, input_idx, input_mask, map_label],
    #                                      outputs = [saliency_conv],
    #                                      on_unused_input='warn')

    f_train_subtask = theano.function(inputs = [image_feat, input_idx, input_mask, map_label],
                              outputs = [map_cost],
                              updates = maps_update_grad,
                              on_unused_input='warn')

    # validation function no gradient updates
    f_val_subtask = theano.function(inputs = [image_feat, input_idx, input_mask, map_label],
                            outputs = [map_cost],
                            on_unused_input='warn')

    f_grad_cache_update_maps, f_param_update_maps \
        = eval(options['optimization'])(shared_params_maps, grad_buf_maps, options)
    logger.info('finished building function')

##############################
#### Pre-Training
##############################

    # calculate how many iterations we need
    num_iters_one_epoch = data_provision_att_vqa_maps.get_size(options['train_split_maps']) / batch_size
    max_iters = options['max_epochs'] * num_iters_one_epoch
    eval_interval_in_iters = options['eval_interval']
    save_interval_in_iters = options['save_interval']
    disp_interval = options['disp_interval']

    best_val_err = 10.0
    best_param = dict()
    beggining_itr = 0
    val_learn_curve_err_map = np.array([])
    train_learn_curve_err_map = np.array([])
    itr_learn_curve = np.array([])
    train_sub_task_x_axis = np.array([])

    for itr in xrange(beggining_itr, max_iters + 1):
        if (itr % eval_interval_in_iters) == 0 or (itr == max_iters):
            val_cost_list = []
            val_map_cost_list = []
            val_accu_list = []
            val_count = 0
            dropout.set_value(numpy.float32(0.))
            for batch_image_feat, batch_question, batch_answer_label, batch_map_label \
                in data_provision_att_vqa_maps.iterate_batch(options['val_split_maps'],
                                                    batch_size):
                input_idx, input_mask \
                    = process_batch(batch_question,
                                    reverse=options['reverse'])
                batch_image_feat = reshape_image_feat(batch_image_feat,
                                                      options['num_region'],
                                                      options['region_dim'])
                [map_cost_val] = f_val_subtask(batch_image_feat, np.transpose(input_idx),
                                     np.transpose(input_mask),
                                     batch_map_label)
                val_count += batch_image_feat.shape[0]
                val_map_cost_list.append(map_cost_val * batch_image_feat.shape[0])

            ave_val_map_cost = sum(val_map_cost_list) / float(val_count)

            if best_val_err > ave_val_map_cost:
                best_val_err = ave_val_map_cost
                shared_to_cpu(shared_params_maps, best_param)
            logger.info('map cost: %f' %(ave_val_map_cost))
            val_learn_curve_err_map = np.append(val_learn_curve_err_map, ave_val_map_cost)
            itr_learn_curve = np.append(itr_learn_curve, itr / float(num_iters_one_epoch))

        dropout.set_value(numpy.float32(1.))


        if options['sample_answer']:
            batch_image_feat, batch_question, batch_answer_label, batch_map_label \
                = data_provision_att_vqa_maps.next_batch_sample(options['train_split_maps'],
                                                       batch_size)
        else:
            batch_image_feat, batch_question, batch_answer_label, batch_map_label \
                = data_provision_att_vqa_maps.next_batch(options['train_split_maps'], batch_size)
        input_idx, input_mask \
            = process_batch(batch_question, reverse=options['reverse'])
        batch_image_feat = reshape_image_feat(batch_image_feat,
                                              options['num_region'],
                                              options['region_dim'])

        [map_cost] = f_train_subtask(batch_image_feat, np.transpose(input_idx),
                               np.transpose(input_mask),
                               batch_map_label)
        # output_norm = f_output_grad_norm()
        # logger.info(output_norm)
        # pdb.set_trace()
        f_grad_clip_subtask()
        f_grad_cache_update_maps()
        lr_t = get_lr(options, itr / float(num_iters_one_epoch))
        f_param_update_maps(lr_t)

        if (itr % eval_interval_in_iters) == 0 or (itr == max_iters):
            train_learn_curve_err_map = np.append(train_learn_curve_err_map, map_cost)
            train_sub_task_x_axis = np.append(train_sub_task_x_axis, itr / float(num_iters_one_epoch))

        if options['shuffle'] and itr > 0 and itr % num_iters_one_epoch == 0:
            data_provision_att_vqa_maps.random_shuffle()

        if (itr % disp_interval) == 0  or (itr == max_iters):
            logger.info('Sub Task: iteration %d/%d epoch %f/%d map_cost %f , lr %f' \
                        % (itr, max_iters,
                           itr / float(num_iters_one_epoch), max_epochs,
                           map_cost, lr_t))

            # [saliency_conv] = f_debug(batch_image_feat, np.transpose(input_idx),
            #                            np.transpose(input_mask),
            #                            batch_map_label)
            if np.isnan(map_cost):
                logger.info('nan detected')
                file_name = options['model_name'] + '_nan_debug.model'
                logger.info('saving the debug model to %s' %(file_name))
                save_model(os.path.join(options['expt_folder'], file_name), options,
                           best_param)
                return 0


    logger.info('best validation error: %f', best_val_err)
    file_name = options['model_name'] + '_best_' + '%.3f' %(best_val_err) + '.model'
    logger.info('saving the best model to %s' %(file_name))
    save_model(os.path.join(options['expt_folder'], file_name), options,
               best_param)


    np.savez_compressed(
        os.path.join(options['expt_folder'], options['model_name']+'_plot_details.npz'),
        valid_error_map=val_learn_curve_err_map,
        x_axis_epochs=itr_learn_curve,
        train_error_map=train_learn_curve_err_map,
        sub_x_axis=train_sub_task_x_axis
    )

    return best_val_err

if __name__ == '__main__':
    logger = log.setup_custom_logger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('changes', nargs='*',
                        help='Changes to default values',
                        default = '')
    args = parser.parse_args()
    for change in args.changes:
        logger.info('dict({%s})'%(change))
        options.update(eval('dict({%s})'%(change)))
    train(options)
