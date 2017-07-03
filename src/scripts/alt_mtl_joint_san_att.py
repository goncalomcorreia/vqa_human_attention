#!/usr/bin/env python

import datetime
import os
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32,exception_verbosity=high"
import sys
import logging
import argparse
import math
sys.path.append('/home/s1670404/vqa_human_attention/src/')
sys.path.append('/home/s1670404/vqa_human_attention/src/data-providers/')
sys.path.append('/home/s1670404/vqa_human_attention/src/models/')
import log
from optimization_weight import *
from joint_mtl_san_att_conv_twolayer_theano import *
from data_provision_att_vqa_with_maps import *
from data_processing_vqa import *

##################
# initialization #
##################
options = OrderedDict()
# data related
options['data_path'] = '/home/s1670404/vqa_human_attention/data_vqa'
options['map_data_path'] = '/home/s1670404/vqa_human_attention/data_att_maps'
options['feature_file'] = 'trainval_feat.h5'
options['expt_folder'] = '/home/s1670404/vqa_human_attention/expt/complete-alt-tasks-mtl'
options['checkpoint_folder'] = os.path.join(options['expt_folder'], 'checkpoints')
options['model_name'] = 'mtl_p_0.5_ce_joint'
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

options['use_kl'] = False
options['task_p'] = 0.5
options['use_second_att_layer'] = False

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
options['load_checkpoints'] = False

# learning parameters
options['optimization'] = 'sgd' # choices
options['batch_size'] = 100
options['lr'] = numpy.float32(0.1)
options['w_emb_lr'] = numpy.float32(80)
options['momentum'] = numpy.float32(0.9)
options['gamma'] = 1
options['step'] = 10
options['step_start'] = 100
options['max_epochs'] = 50
options['weight_decay'] = 5e-4
options['weight_decay_sub'] = 5e-4
options['decay_rate'] = numpy.float32(0.999)
options['drop_ratio'] = numpy.float32(0.5)
options['smooth'] = numpy.float32(1e-8)
options['grad_clip'] = numpy.float32(0.1)

# log params
options['disp_interval'] = 10
options['eval_interval'] = 500
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

    if not os.path.exists(options['checkpoint_folder']):
        os.makedirs(options['checkpoint_folder'])

    checkpoint_model_path = os.path.join(options['checkpoint_folder'], options['model_name'])
    if not os.path.exists(checkpoint_model_path):
        os.makedirs(checkpoint_model_path)

    if len(os.listdir(checkpoint_model_path))>0:
        n_shuffles = pickle.load(open(os.path.join(checkpoint_model_path, 'n_shuffles.p'), "rb" ))
        state = pickle.load(open(os.path.join(checkpoint_model_path, 'state.p'), "rb" ))
        data_provision_att_vqa = DataProvisionAttVqaTest(options['data_path'],
                                                     options['feature_file'],
                                                     n_shuffles=n_shuffles,
                                                     state=state)
        data_provision_att_vqa_maps = DataProvisionAttVqaWithMaps(options['data_path'],
                                                                  options['feature_file'],
                                                                  options['map_data_path'],
                                                                  n_shuffles=n_shuffles,
                                                                  state=state)
    else:
        data_provision_att_vqa = DataProvisionAttVqaTest(options['data_path'],
                                                     options['feature_file'])
        data_provision_att_vqa_maps = DataProvisionAttVqaWithMaps(options['data_path'],
                                                                  options['feature_file'],
                                                                  options['map_data_path'])

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']

    ###############
    # build model #
    ###############
    if len(os.listdir(checkpoint_model_path))>0 and options['load_checkpoints']:
        logger.info('Checkpoint files found!')
        logger.info('Loading checkpoint files...')
        best_param = dict()
        for check_file in os.listdir(checkpoint_model_path):
            check_model_path = os.path.join(checkpoint_model_path, check_file)
            if 'best' in check_file:
                options_best, params_best, shared_params_best = load_model(check_model_path)
                best_val_accu = float('.'.join(check_file.split('_')[-1].split('.')[0:-1]))
                shared_to_cpu(shared_params_best, best_param)
            elif 'checkpoint' in check_file:
                options, params, shared_params = load_model(check_model_path)
                beggining_itr = int(check_file.split('_')[-1].split('.')[0])

        plot_details_path = os.path.join(checkpoint_model_path, options['model_name']+'_plot_details.npz')

        with np.load(plot_details_path) as data:
            val_learn_curve_acc = data['valid_accuracy']
            val_learn_curve_err = data['valid_error']
            val_learn_curve_err_map = data['valid_error_map']
            train_learn_curve_err_map = data['train_error_map']
            itr_learn_curve = data['x_axis_epochs']
            train_learn_curve_err = data['train_error']
            train_learn_curve_err = data['train_accuracy']

    else:
        params = init_params(options)
        shared_params = init_shared_params(params)

    shared_params_maps = init_shared_params_maps(shared_params, options)

    image_feat, input_idx, input_mask, \
        label, dropout, ans_cost, accu, pred_label, \
        prob_attention_1, prob_attention_2, map_cost, map_label, total_cost = build_model(shared_params, options)

    logger.info('finished building model')

    ####################
    # add weight decay #
    ####################
    weight_decay = theano.shared(numpy.float32(options['weight_decay']),\
                                 name = 'weight_decay')
    weight_decay_sub = theano.shared(numpy.float32(options['weight_decay_sub']),\
                                 name = 'weight_decay_sub')

    reg_cost = 0

    for k in shared_params.iterkeys():
        if k != 'w_emb':
            reg_cost += (shared_params[k]**2).sum()

    reg_map = 0

    for k in shared_params_maps.iterkeys():
        if k != 'w_emb':
            reg_map += (shared_params_maps[k]**2).sum()

    reg_cost *= weight_decay
    reg_map *= weight_decay_sub

    ans_reg_cost = ans_cost + reg_cost
    map_reg_cost = total_cost + reg_map

    ###############
    # # gradients #
    ###############
    ans_grads = T.grad(ans_reg_cost, wrt = shared_params.values())
    map_grads = T.grad(map_reg_cost, wrt = shared_params_maps.values())

    grad_buf_maps = [theano.shared(p.get_value() * 0, name='%s_grad_buf' % k )
                     for k, p in shared_params_maps.iteritems()]
    grad_buf = []
    for elem in grad_buf_maps:
        grad_buf.append(elem)
    for k, p in shared_params.iteritems():
        if k not in shared_params_maps.keys():
            grad_buf.append(theano.shared(p.get_value() * 0, name='%s_grad_buf' % k ))

    # accumulate the gradients within one batch
    ans_update_grad = [(g_b, g) for g_b, g in zip(grad_buf, ans_grads)]
    maps_update_grad = [(g_b, g) for g_b, g in zip(grad_buf_maps, map_grads)]
    # need to declare a share variable ??
    grad_clip = options['grad_clip']
    grad_norm = [T.sqrt(T.sum(g_b**2)) for g_b in grad_buf]
    update_clip = [(g_b, T.switch(T.gt(g_norm, grad_clip),
                                  g_b*grad_clip/g_norm, g_b))
                   for (g_norm, g_b) in zip(grad_norm, grad_buf)]

    grad_norm_maps = [T.sqrt(T.sum(g_b**2)) for g_b in grad_buf_maps]
    update_clip_maps = [(g_b, T.switch(T.gt(g_norm, grad_clip),
                                  g_b*grad_clip/g_norm, g_b))
                   for (g_norm, g_b) in zip(grad_norm_maps, grad_buf_maps)]

    # corresponding update function
    f_grad_clip = theano.function(inputs = [],
                                  updates = update_clip)
    f_output_grad_norm = theano.function(inputs = [],
                                         outputs = grad_norm)
    f_grad_clip_subtask = theano.function(inputs = [],
                                  updates = update_clip_maps)
    f_output_grad_norm_subtask = theano.function(inputs = [],
                                         outputs = grad_norm_maps)
    f_train = theano.function(inputs = [image_feat, input_idx, input_mask, label],
                              outputs = [ans_cost, accu],
                              updates = ans_update_grad,
                              on_unused_input='warn')

    f_train_subtask = theano.function(inputs = [image_feat, input_idx, input_mask, label, map_label],
                              outputs = [total_cost, map_cost],
                              updates = maps_update_grad,
                              on_unused_input='warn')

    # validation function no gradient updates
    f_val = theano.function(inputs = [image_feat, input_idx, input_mask, label],
                            outputs = [ans_cost, accu],
                            on_unused_input='warn')
    f_val_subtask = theano.function(inputs = [image_feat, input_idx, input_mask, map_label],
                            outputs = [map_cost],
                            on_unused_input='warn')

    f_grad_cache_update, f_param_update \
        = eval(options['optimization'])(shared_params, grad_buf, options)
    f_grad_cache_update_maps, f_param_update_maps \
        = eval(options['optimization'])(shared_params_maps, grad_buf_maps, options)
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
        val_learn_curve_err_map = np.array([])
        train_learn_curve_err_map = np.array([])
        itr_learn_curve = np.array([])
        train_learn_curve_err = np.array([])
        train_learn_curve_acc = np.array([])
        train_main_task_x_axis = np.array([])
        train_sub_task_x_axis = np.array([])

    checkpoint_param = dict()
    checkpoint_iter_interval = num_iters_one_epoch

    # Make sure always using the same random seed
    logger.info("Picked a seed!")
    rng = np.random.RandomState(1234)

    for itr in xrange(beggining_itr, max_iters + 1):
        if (itr % eval_interval_in_iters) == 0 or (itr == max_iters):
            val_cost_list = []
            val_map_cost_list = []
            val_accu_list = []
            val_count = 0
            dropout.set_value(numpy.float32(0.))
            for batch_image_feat, batch_question, batch_answer_label, batch_map_label \
                in data_provision_att_vqa_maps.iterate_batch(options['val_split'],
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
            val_count = 0
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
            logger.info('validation cost: %f accu: %f map cost: %f' %(ave_val_cost, ave_val_accu, ave_val_map_cost))
            val_learn_curve_acc = np.append(val_learn_curve_acc, ave_val_accu)
            val_learn_curve_err = np.append(val_learn_curve_err, ave_val_cost)
            val_learn_curve_err_map = np.append(val_learn_curve_err_map, ave_val_map_cost)
            itr_learn_curve = np.append(itr_learn_curve, itr / float(num_iters_one_epoch))

        if (itr % checkpoint_iter_interval) == 0:
            shared_to_cpu(shared_params, checkpoint_param)
            if itr>0:
                previous_itr = itr - checkpoint_iter_interval
                checkpoint_model = options['model_name'] + '_checkpoint_' + '%d' %(previous_itr) + '.model'
                if checkpoint_model in os.listdir(checkpoint_model_path):
                    os.remove(os.path.join(checkpoint_model_path, checkpoint_model))
            file_name = options['model_name'] + '_checkpoint_' + '%d' %(itr) + '.model'
            logger.info('saving a checkpoint model to %s' %(file_name))
            save_model(os.path.join(checkpoint_model_path, file_name), options,
                       checkpoint_param)
            for checkpoint_model in os.listdir(checkpoint_model_path):
                if 'best' in checkpoint_model:
                    os.remove(os.path.join(checkpoint_model_path, checkpoint_model))
            logger.info('best validation accu so far: %f', best_val_accu)
            file_name = options['model_name'] + '_best_' + '%.3f' %(best_val_accu) + '.model'
            logger.info('saving the best model so far to %s' %(file_name))
            save_model(os.path.join(checkpoint_model_path, file_name), options,
                       best_param)
            np.savez_compressed(
                os.path.join(checkpoint_model_path, options['model_name']+'_plot_details.npz'),
                valid_error_map=val_learn_curve_err_map,
                valid_error=val_learn_curve_err,
                valid_accuracy=val_learn_curve_acc,
                x_axis_epochs=itr_learn_curve,
                train_error=train_learn_curve_err,
                train_accuracy=train_learn_curve_acc,
                train_error_map=train_learn_curve_err_map,
                main_x_axis=train_main_task_x_axis,
                sub_x_axis=train_sub_task_x_axis
            )

            n_shuffles = int(itr / float(num_iters_one_epoch))-1
            state = data_provision_att_vqa.rng.get_state()

            for checkpoint_file in os.listdir(checkpoint_model_path):
                if 'shuffles' in checkpoint_file or 'state' in checkpoint_file:
                    os.remove(os.path.join(checkpoint_model_path, checkpoint_file))

            pickle.dump(n_shuffles, open(os.path.join(checkpoint_model_path, 'n_shuffles.p'), "wb" ))
            pickle.dump(state, open(os.path.join(checkpoint_model_path, 'state.p'), "wb" ))

        dropout.set_value(numpy.float32(1.))

        task_choice = rng.choice(2, p=[1-options['task_p'], options['task_p']])

        if task_choice==1:

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
        else:
            if options['sample_answer']:
                batch_image_feat, batch_question, batch_answer_label, batch_map_label \
                    = data_provision_att_vqa_maps.next_batch_sample(options['train_split'],
                                                           batch_size)
            else:
                batch_image_feat, batch_question, batch_answer_label, batch_map_label \
                    = data_provision_att_vqa_maps.next_batch(options['train_split'], batch_size)
            input_idx, input_mask \
                = process_batch(batch_question, reverse=options['reverse'])
            batch_image_feat = reshape_image_feat(batch_image_feat,
                                                  options['num_region'],
                                                  options['region_dim'])

            [total_cost, map_cost] = f_train_subtask(batch_image_feat,
                                                     np.transpose(input_idx),
                                                     np.transpose(input_mask),
                                                     batch_map_label)

        if task_choice==1:
            f_grad_clip()
            f_grad_cache_update()
            lr_t = get_lr(options, itr / float(num_iters_one_epoch))
            f_param_update(lr_t)
        else:
            f_grad_clip_subtask()
            f_grad_cache_update_maps()
            lr_t = get_lr(options, itr / float(num_iters_one_epoch))
            f_param_update_maps(lr_t)

        if (itr % eval_interval_in_iters) == 0 or (itr == max_iters):
            if task_choice==1:
                train_learn_curve_err = np.append(train_learn_curve_err, cost)
                train_learn_curve_acc = np.append(train_learn_curve_acc, accu)
                train_main_task_x_axis = np.append(train_main_task_x_axis, itr / float(num_iters_one_epoch))
            else:
                train_learn_curve_err_map = np.append(train_learn_curve_err_map, map_cost)
                train_sub_task_x_axis = np.append(train_sub_task_x_axis, itr / float(num_iters_one_epoch))

        if options['shuffle'] and itr > 0 and itr % num_iters_one_epoch == 0:
            logger.info("Everyday I'm shuffling!")
            data_provision_att_vqa.random_shuffle()
            data_provision_att_vqa_maps.random_shuffle()

        if (itr % disp_interval) == 0  or (itr == max_iters):
            if task_choice==1:
                logger.info('Main Task: iteration %d/%d epoch %f/%d cost %f accu %f, lr %f' \
                            % (itr, max_iters,
                               itr / float(num_iters_one_epoch), max_epochs,
                               cost, accu, lr_t))
            else:
                logger.info('Sub Task: iteration %d/%d epoch %f/%d map_cost %f , total_cost %f, lr %f' \
                            % (itr, max_iters,
                               itr / float(num_iters_one_epoch), max_epochs,
                               map_cost, total_cost, lr_t))
            if 'cost' in locals():
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

    if len(os.listdir(checkpoint_model_path))>0:
        logger.info('Deleting checkpoint files...')
        for check_file in os.listdir(checkpoint_model_path):
            os.remove(os.path.join(checkpoint_model_path, check_file))

    np.savez_compressed(
        os.path.join(options['expt_folder'], options['model_name']+'_plot_details.npz'),
        valid_error_map=val_learn_curve_err_map,
        valid_error=val_learn_curve_err,
        valid_accuracy=val_learn_curve_acc,
        x_axis_epochs=itr_learn_curve,
        train_error=train_learn_curve_err,
        train_accuracy=train_learn_curve_acc,
        train_error_map=train_learn_curve_err_map,
        main_x_axis=train_main_task_x_axis,
        sub_x_axis=train_sub_task_x_axis
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
