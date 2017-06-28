#!/usr/bin/env python

import datetime
import os
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
import sys
import logging
import argparse
import pickle
import math
sys.path.append('/home/s1670404/imageqa-san/src/')
sys.path.append('/home/s1670404/vqa_human_attention/src/data-providers/')
sys.path.append('/home/s1670404/vqa_human_attention/src/models/')
import log
from optimization_weight import *
from san_att_conv_twolayer_theano import *
from data_provision_att_vqa_test import DataProvisionAttVqaTest
from data_processing_vqa import *

##################
# initialization #
##################
options = OrderedDict()
# data related
options['data_path'] = '/home/s1670404/vqa_human_attention/data_vqa'
options['feature_file'] = 'trainval_feat.h5'
options['expt_folder'] = '/home/s1670404/vqa_human_attention/expt/baseline'
options['checkpoint_folder'] = os.path.join(options['expt_folder'], 'checkpoints')
options['model_name'] = 'baseline'
options['train_split'] = 'trainval1'
options['val_split'] = 'val2'
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
options['optimization'] = 'sgd' # sgd
options['batch_size'] = 100
options['lr'] = numpy.float32(0.1)
options['w_emb_lr'] = numpy.float32(80)
options['momentum'] = numpy.float32(0.9)
options['gamma'] = 1
options['step'] = 10
options['step_start'] = 100
options['max_epochs'] = 50
options['weight_decay'] = 5e-4
options['decay_rate'] = numpy.float32(0.999)
options['drop_ratio'] = numpy.float32(0.5)
options['smooth'] = numpy.float32(1e-8)
options['grad_clip'] = numpy.float32(0.1)

# log params
options['disp_interval'] = 10
options['eval_interval'] = 1000
options['save_interval'] = 500

def get_lr(options, curr_epoch):
    if options['optimization'] == 'sgd':
        power = max((curr_epoch - options['step_start']) / options['step'], 0)
        power = math.ceil(power)
        return options['lr'] * (options['gamma'] ** power)  #
    else:
        return options['lr']

def train(options):

    logger = logging.getLogger('root')
    logger.info(options)
    logger.info('start training')

    if not os.path.exists(options['checkpoint_folder']):
        os.makedirs(options['checkpoint_folder'])

    if len(os.listdir(options['checkpoint_folder']))>0:
        n_shuffles = pickle.load(open(os.path.join(options['checkpoint_folder'], 'n_shuffles.p'), "rb" ))
        state = pickle.load(open(os.path.join(options['checkpoint_folder'], 'state.p'), "rb" ))
        data_provision_att_vqa = DataProvisionAttVqaTest(options['data_path'],
                                                     options['feature_file'],
                                                     n_shuffles=n_shuffles,
                                                     state=state)
    else:
        data_provision_att_vqa = DataProvisionAttVqaTest(options['data_path'],
                                                     options['feature_file'])

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']

    ###############
    # build model #
    ###############

    if len(os.listdir(options['checkpoint_folder']))>0:
        logger.info('Checkpoint files found!')
        logger.info('Loading checkpoint files...')
        best_param = dict()
        for check_file in os.listdir(options['checkpoint_folder']):
            check_model_path = os.path.join(options['checkpoint_folder'], check_file)
            if 'best' in check_file:
                options_best, params_best, shared_params_best = load_model(check_model_path)
                best_val_accu = float('.'.join(check_file.split('_')[-1].split('.')[0:-1]))
                shared_to_cpu(shared_params_best, best_param)
            elif 'checkpoint' in check_file:
                options, params, shared_params = load_model(check_model_path)
                beggining_itr = int(check_file.split('_')[-1].split('.')[0])

        plot_details_path = os.path.join(options['checkpoint_folder'], options['model_name']+'_plot_details.npz')

        with np.load(plot_details_path) as data:
            val_learn_curve_acc = data['valid_accuracy']
            val_learn_curve_err = data['valid_error']
            itr_learn_curve = data['x_axis_epochs']
            train_learn_curve_err = data['train_error']
            train_learn_curve_acc = data['train_accuracy']

    else:
        params = init_params(options)
        shared_params = init_shared_params(params)

    image_feat, input_idx, input_mask, \
        label, dropout, cost, accu, pred_label, \
        prob_attention_1, prob_attention_2 \
        = build_model(shared_params, options)

    logger.info('finished building model')

    ####################
    # add weight decay #
    ####################
    weight_decay = theano.shared(numpy.float32(options['weight_decay']),\
                                 name = 'weight_decay')
    reg_cost = 0

    for k in shared_params.iterkeys():
        if k != 'w_emb':
            reg_cost += (shared_params[k]**2).sum()

    reg_cost *= weight_decay
    reg_cost = cost + reg_cost

    ###############
    # # gradients #
    ###############
    grads = T.grad(reg_cost, wrt = shared_params.values())
    grad_buf = [theano.shared(p.get_value() * 0, name='%s_grad_buf' % k )
                for k, p in shared_params.iteritems()]
    # accumulate the gradients within one batch
    update_grad = [(g_b, g) for g_b, g in zip(grad_buf, grads)]
    # need to declare a share variable ??
    grad_clip = options['grad_clip']
    grad_norm = [T.sqrt(T.sum(g_b**2)) for g_b in grad_buf]
    update_clip = [(g_b, T.switch(T.gt(g_norm, grad_clip),
                                  g_b*grad_clip/g_norm, g_b))
                   for (g_norm, g_b) in zip(grad_norm, grad_buf)]

    # corresponding update function
    f_grad_clip = theano.function(inputs = [],
                                  updates = update_clip)
    f_output_grad_norm = theano.function(inputs = [],
                                         outputs = grad_norm)
    f_train = theano.function(inputs = [image_feat, input_idx, input_mask, label],
                              outputs = [cost, accu],
                              updates = update_grad,
                              on_unused_input='warn')
    # validation function no gradient updates
    f_val = theano.function(inputs = [image_feat, input_idx, input_mask, label],
                            outputs = [cost, accu],
                            on_unused_input='warn')

    f_grad_cache_update, f_param_update \
        = eval(options['optimization'])(shared_params, grad_buf, options)
    logger.info('finished building function')

    # calculate how many iterations we need
    num_iters_one_epoch = data_provision_att_vqa.get_size(options['train_split']) / batch_size
    max_iters = max_epochs * num_iters_one_epoch
    eval_interval_in_iters = options['eval_interval']
    save_interval_in_iters = options['save_interval']
    disp_interval = options['disp_interval']

    if 'best_val_accu' not in locals():
        best_val_accu = 0.0
        best_param = dict()
        beggining_itr = 0
        val_learn_curve_acc = np.array([])
        val_learn_curve_err = np.array([])
        itr_learn_curve = np.array([])
        train_learn_curve_err = np.array([])
        train_learn_curve_acc = np.array([])

    checkpoint_param = dict()
    checkpoint_iter_interval = num_iters_one_epoch

    for itr in xrange(beggining_itr, max_iters + 1):
        if (itr % eval_interval_in_iters) == 0 or (itr == max_iters):
            val_cost_list = []
            val_accu_list = []
            val_count = 0
            dropout.set_value(numpy.float32(0.))
            for batch_image_feat, batch_question, batch_answer_label \
                in data_provision_att_vqa.iterate_batch(options['val_split'],
                                                    batch_size):
                input_idx, input_mask \
                    = process_batch(batch_question,
                                    reverse=options['reverse'])
                batch_image_feat = reshape_image_feat(batch_image_feat,
                                                      options['num_region'],
                                                      options['region_dim'])
                [cost, accu] = f_val(batch_image_feat, np.transpose(input_idx),
                                     np.transpose(input_mask),
                                     batch_answer_label.astype('int32').flatten())
                val_count += batch_image_feat.shape[0]
                val_cost_list.append(cost * batch_image_feat.shape[0])
                val_accu_list.append(accu * batch_image_feat.shape[0])
            ave_val_cost = sum(val_cost_list) / float(val_count)
            ave_val_accu = sum(val_accu_list) / float(val_count)
            if best_val_accu < ave_val_accu:
                best_val_accu = ave_val_accu
                shared_to_cpu(shared_params, best_param)
            logger.info('validation cost: %f accu: %f' %(ave_val_cost, ave_val_accu))
            val_learn_curve_acc = np.append(val_learn_curve_acc, ave_val_accu)
            val_learn_curve_err = np.append(val_learn_curve_err, ave_val_cost)
            itr_learn_curve = np.append(itr_learn_curve, itr / float(num_iters_one_epoch))

        if (itr % checkpoint_iter_interval) == 0:
            shared_to_cpu(shared_params, checkpoint_param)
            if itr>0:
                previous_itr = itr - checkpoint_iter_interval
                checkpoint_model = options['model_name'] + '_checkpoint_' + '%d' %(previous_itr) + '.model'
                if checkpoint_model in os.listdir(options['checkpoint_folder']):
                    os.remove(os.path.join(options['checkpoint_folder'], checkpoint_model))
            file_name = options['model_name'] + '_checkpoint_' + '%d' %(itr) + '.model'
            logger.info('saving a checkpoint model to %s' %(file_name))
            save_model(os.path.join(options['checkpoint_folder'], file_name), options,
                       checkpoint_param)
            for checkpoint_model in os.listdir(options['checkpoint_folder']):
                if 'best' in checkpoint_model:
                    os.remove(os.path.join(options['checkpoint_folder'], checkpoint_model))
            logger.info('best validation accu so far: %f', best_val_accu)
            file_name = options['model_name'] + '_best_' + '%.3f' %(best_val_accu) + '.model'
            logger.info('saving the best model so far to %s' %(file_name))
            save_model(os.path.join(options['checkpoint_folder'], file_name), options,
                       best_param)
            np.savez_compressed(
                os.path.join(options['checkpoint_folder'], options['model_name']+'_plot_details.npz'),
                valid_error=val_learn_curve_err,
                valid_accuracy=val_learn_curve_acc,
                x_axis_epochs=itr_learn_curve,
                train_error=train_learn_curve_err,
                train_accuracy=train_learn_curve_acc
            )

            n_shuffles = int(itr / float(num_iters_one_epoch))-1
            state = data_provision_att_vqa.rng.get_state()

            for checkpoint_file in os.listdir(options['checkpoint_folder']):
                if 'shuffles' in checkpoint_file or 'state' in checkpoint_file:
                    os.remove(os.path.join(options['checkpoint_folder'], checkpoint_file))

            pickle.dump(n_shuffles, open(os.path.join(options['checkpoint_folder'], 'n_shuffles.p'), "wb" ))
            pickle.dump(state, open(os.path.join(options['checkpoint_folder'], 'state.p'), "wb" ))

        dropout.set_value(numpy.float32(1.))
        if options['sample_answer']:
            batch_image_feat, batch_question, batch_answer_label \
                = data_provision_att_vqa.next_batch_sample(options['train_split'],
                                                       batch_size)
        else:
            batch_image_feat, batch_question, batch_answer_label \
                = data_provision_att_vqa.next_batch(options['train_split'], batch_size)
        input_idx, input_mask \
            = process_batch(batch_question, reverse=options['reverse'])
        batch_image_feat = reshape_image_feat(batch_image_feat,
                                              options['num_region'],
                                              options['region_dim'])

        [cost, accu] = f_train(batch_image_feat, np.transpose(input_idx),
                               np.transpose(input_mask),
                               batch_answer_label.astype('int32').flatten())
        # output_norm = f_output_grad_norm()
        # logger.info(output_norm)
        # pdb.set_trace()
        f_grad_clip()
        f_grad_cache_update()
        lr_t = get_lr(options, itr / float(num_iters_one_epoch))
        f_param_update(lr_t)

        if (itr % eval_interval_in_iters) == 0 or (itr == max_iters):
            train_learn_curve_err = np.append(train_learn_curve_err, cost)
            train_learn_curve_acc = np.append(train_learn_curve_acc, accu)

        if options['shuffle'] and itr > 0 and itr % num_iters_one_epoch == 0:
            logger.info("Everyday I'm shuffling!")
            data_provision_att_vqa.random_shuffle()

        if (itr % disp_interval) == 0  or (itr == max_iters):
            logger.info('iteration %d/%d epoch %f/%d cost %f accu %f, lr %f' \
                        % (itr, max_iters,
                           itr / float(num_iters_one_epoch), max_epochs,
                           cost, accu, lr_t))
            if np.isnan(cost):
                logger.info('nan detected')
                file_name = options['model_name'] + '_nan_debug.model'
                logger.info('saving the debug model to %s' %(file_name))
                save_model(os.path.join(options['expt_folder'], file_name), options,
                           best_param)
                return 0


    logger.info('best validation accu: %f', best_val_accu)
    file_name = options['model_name'] + '_best_' + '%.3f' %(best_val_accu) + '.model'
    logger.info('saving the best model to %s' %(file_name))
    save_model(os.path.join(options['expt_folder'], file_name), options,
               best_param)

    if len(os.listdir(options['checkpoint_folder']))>0:
        logger.info('Deleting checkpoint files...')
        for check_file in os.listdir(options['checkpoint_folder']):
            os.remove(os.path.join(options['checkpoint_folder'], check_file))

    np.savez_compressed(
        os.path.join(options['expt_folder'], options['model_name']+'_plot_details.npz'),
        valid_error=val_learn_curve_err,
        valid_accuracy=val_learn_curve_acc,
        x_axis_epochs=itr_learn_curve,
        train_error=train_learn_curve_err,
        train_accuracy=train_learn_curve_acc
    )

    return best_val_accu

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
