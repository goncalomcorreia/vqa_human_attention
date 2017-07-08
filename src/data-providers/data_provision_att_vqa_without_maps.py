#!/usr/bin/env python

from data_provision_att_vqa import *
import time

logger = logging.getLogger('root')

class DataProvisionAttVqaWithoutMaps(DataProvisionAttVqa):
    def __init__(self, data_folder, feature_file, maps_data_folder, rng = None, state = None, n_shuffles = None):
        self._image_feat = self.load_image_feat(data_folder, feature_file)
        self._question_id = OrderedDict()
        self._image_id = OrderedDict()
        self._question = OrderedDict()
        # answer set
        self._answer = OrderedDict()
        # answer counter
        self._answer_counter = OrderedDict()
        # most common answer
        self._answer_label = OrderedDict()
        self._splits = ['train', 'val1', 'val2', 'val2_all']
        self._pointer = OrderedDict()
        if rng is None:
            rng = np.random.RandomState(1234)
        self.rng = rng
        for split in self._splits:
            with open(os.path.join(data_folder, split) + '.pkl') as f:
                split_question_id = pkl.load(f)
                split_image_id = pkl.load(f)
                split_question = pkl.load(f)
                split_answer = pkl.load(f)
                split_answer_counter = pkl.load(f)
                split_answer_label = pkl.load(f)
            idx = range(split_question.shape[0])
            if not (split is 'val2' or split is 'val2_all'):
                idx = self.rng.permutation(split_question.shape[0])
            self._question_id[split] = split_question_id[idx]
            self._image_id[split] = split_image_id[idx]
            self._question[split] = split_question[idx]
            self._answer[split] = split_answer[idx]
            self._answer_counter[split] = split_answer_counter[idx]
            self._answer_label[split] = split_answer_label[idx]
            self._pointer[split] = 0
        self._splits.append('trainval1')
        self._pointer['trainval1'] = 0

        self._att_maps = OrderedDict()
        self._att_maps_qids = OrderedDict()
        maps = h5py.File(os.path.join(maps_data_folder,'map_dist_196.h5'), 'r')
        att_maps = np.array(maps['label'])
        maps.close()
        for maps_split in ['train', 'val']:
            with open(os.path.join(maps_data_folder, maps_split) + '.pkl') as f:
                self._att_maps_qids[maps_split] = pkl.load(f)
            if maps_split == 'train':
                self._att_maps[maps_split] = att_maps[:len(self._att_maps_qids[maps_split])]
            else:
                self._att_maps[maps_split] = att_maps[len(self._att_maps_qids[maps_split]):]
        self._map_label = OrderedDict()
        qids_with_maps = self._att_maps_qids['train'] + self._att_maps_qids['val']
        for split in self._splits[:-1]:
            if split=='train':
                maps_split = split
            else:
                maps_split = 'val'
            map_idx = np.where(np.in1d(self._question_id[split], self._att_maps_qids[maps_split]))[0]
            self._question_id[split] = np.delete(self._question_id[split],map_idx,axis=0)
            self._image_id[split] = np.delete(self._image_id[split],map_idx,axis=0)
            self._question[split] = np.delete(self._question[split],map_idx,axis=0)
            self._answer[split] = np.delete(self._answer[split],map_idx,axis=0)
            self._answer_counter[split] = np.delete(self._answer_counter[split],map_idx,axis=0)
            self._answer_label[split] = np.delete(self._answer_label[split],map_idx,axis=0)

        self._question_id['trainval1'] = np.concatenate([self._question_id['train'],
                                                         self._question_id['val1']],
                                                        axis = 0)
        self._image_id['trainval1'] = np.concatenate([self._image_id['train'],
                                                      self._image_id['val1']],
                                                     axis = 0)
        self._question['trainval1'] = np.concatenate([self._question['train'],
                                                      self._question['val1']],
                                                     axis = 0)
        self._answer['trainval1'] = np.concatenate([self._answer['train'],
                                                    self._answer['val1']],
                                                   axis = 0)
        self._answer_counter['trainval1'] \
            = np.concatenate([self._answer_counter['train'],
                              self._answer_counter['val1']],
                             axis = 0)
        self._answer_label['trainval1'] \
            = np.concatenate([self._answer_label['train'],
                              self._answer_label['val1']],
                             axis = 0)

        logger.info('finished loading vqa data')

        if n_shuffles is not None:
            for i in xrange(n_shuffles):
                self.random_shuffle()

        if state is not None:
            self.rng.set_state(state)
