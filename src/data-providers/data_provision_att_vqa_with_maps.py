#!/usr/bin/env python

from data_provision_att_vqa_test import *
import time

logger = logging.getLogger('root')

class DataProvisionAttVqaWithMaps(DataProvisionAttVqaTest):
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
        aux_att_maps = []
        self._att_maps_qids['val'] = list(set(self._att_maps_qids['val']))
        for i, elem in enumerate(self._att_maps['val']):
            if i%3==0:
                aux_att_maps.append(elem)
        self._att_maps['val'] = np.array(aux_att_maps)
        self._map_label = OrderedDict()
        qids_with_maps = self._att_maps_qids['train'] + self._att_maps_qids['val']
        # Remove human attention maps that are all zero
        qids_with_maps = np.delete(np.array(qids_with_maps),
                             np.array(list(set(np.where(att_maps==0)[0]))),
                             axis=0)
        qids_with_maps = list(qids_with_maps)
        att_maps = att_maps[~np.all(att_maps == 0, axis=1)]
        for split in self._splits:
            map_idx = np.where(np.in1d(self._question_id[split], qids_with_maps))[0]
            self._question_id[split] = self._question_id[split][map_idx]
            self._image_id[split] = self._image_id[split][map_idx]
            self._question[split] = self._question[split][map_idx]
            self._answer[split] = self._answer[split][map_idx]
            self._answer_counter[split] = self._answer_counter[split][map_idx]
            self._answer_label[split] = self._answer_label[split][map_idx]
            sort = np.argsort(qids_with_maps)
            rank = np.searchsorted(qids_with_maps, self._question_id[split], sorter=sort)
            self._map_label[split] = att_maps[np.array(sort[rank])]

        if n_shuffles is not None:
            for i in xrange(n_shuffles):
                self.random_shuffle()

        if state is not None:
            self.rng.set_state(state)

    def get_map_from_qid(self, qid):
        for maps_split in ['train', 'val']:
            if qid in self._att_maps_qids[maps_split]:
                idx = self._att_maps_qids[maps_split].index(qid)
                return self._att_maps[maps_split][idx]
        return -1

    def random_shuffle(self):
        for split in self._splits:
            if not (split is 'val2' or split is 'val2_all'):
                idx = range(len(self._question[split]))
                random.shuffle(idx)
                idx = self.rng.permutation(len(self._question[split]))
                self._question_id[split] = self._question_id[split][idx]
                self._image_id[split] = self._image_id[split][idx]
                self._question[split] = self._question[split][idx]
                self._answer[split] = self._answer[split][idx]
                self._answer_counter[split] = self._answer_counter[split][idx]
                self._answer_label[split] = self._answer_label[split][idx]
                self._pointer[split] = 0
                self._map_label[split] = self._map_label[split][idx]

    def iterate_batch(self, partition, batch_size):
        logger.debug('begin to iterate batch for %s'%(partition))
        current = 0
        while current + batch_size <= self._question[partition].shape[0]:
            batch_image_id = self._image_id[partition][current :
                                                       current + batch_size]
            batch_question = self._question[partition][current :
                                                       current + batch_size]
            # index - 1 as query for image feature
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()

            batch_answer_label = self._answer_label[partition][current :
                                                               current + batch_size]
            batch_map_label = self._map_label[partition][current :
                                                               current + batch_size]
            yield batch_image_feat, batch_question, batch_answer_label, batch_map_label
            current = current + batch_size
            logger.debug('iterating batch at current: %d'%(current))
        if current != self._question[partition].shape[0]:
            batch_image_id = self._image_id[partition][current :]
            batch_question = self._question[partition][current :]
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()
            batch_answer_label = self._answer_label[partition][current :]
            batch_map_label = self._map_label[partition][current :]
            logger.debug('finished iterating batch for %s'%(partition))
            yield batch_image_feat, batch_question, batch_answer_label, batch_map_label

    def next_batch(self, partition, batch_size):
        if self._pointer[partition] + batch_size <= self._question[partition].shape[0]:
            batch_question = self._question[partition][self._pointer[partition] :
                                                       self._pointer[partition]
                                                       + batch_size]
            batch_image_id = self._image_id[partition][self._pointer[partition] :
                                                       self._pointer[partition]
                                                       + batch_size]
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()

            batch_answer_label = self._answer_label[partition][self._pointer[partition] :
                                                               self._pointer[partition]
                                                               + batch_size]
            batch_map_label = self._map_label[partition][self._pointer[partition] :
                                                               self._pointer[partition]
                                                               + batch_size]
            # update pointer
            self._pointer[partition] = (self._pointer[partition] + batch_size) \
                                       % self._question[partition].shape[0]
            logger.debug('next batch at pointer: %d'%(self._pointer[partition]))
            return batch_image_feat, batch_question, batch_answer_label, batch_map_label
        else:
            logger.debug('new epoch of data iteration')
            next_pointer = (self._pointer[partition] + batch_size) \
                           % self._question[partition].shape[0]
            batch_question = self._question[partition][self._pointer[partition]:]
            batch_question = np.append(batch_question,
                                       self._question[partition][:next_pointer],
                                       axis = 0)
            batch_image_id = self._image_id[partition][self._pointer[partition]:]
            batch_image_id = np.append(batch_image_id,
                                       self._image_id[partition][:next_pointer],
                                       axis = 0)

            # index - 1 as query for image feature
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()

            batch_answer_label = self._answer_label[partition][self._pointer[partition]:]
            batch_answer_label = np.append(batch_answer_label,
                                           self._answer_label[partition][:next_pointer],
                                           axis = 0)
            batch_map_label = np.append(batch_map_label,
                                           self._map_label[partition][:next_pointer],
                                           axis = 0)
            self._pointer[partition] = next_pointer
            logger.debug('next batch at pointer: %d'%(next_pointer))
            return batch_image_feat, batch_question, batch_answer_label, batch_map_label

    def next_batch_sample(self, partition, batch_size):
        if self._pointer[partition] + batch_size <= self._question[partition].shape[0]:
            batch_question = self._question[partition][self._pointer[partition] :
                                                       self._pointer[partition]
                                                       + batch_size]
            batch_image_id = self._image_id[partition][self._pointer[partition] :
                                                       self._pointer[partition]
                                                       + batch_size]
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()
            batch_answer = self._answer[partition][self._pointer[partition]:
                                                   self._pointer[partition]
                                                   + batch_size]
            batch_answer_label = [self.rng.choice(ans)for ans in batch_answer]
            batch_answer_label = np.array(batch_answer_label)
            batch_map_label = self._map_label[partition][self._pointer[partition] :
                                                               self._pointer[partition]
                                                               + batch_size]
            # update pointer
            self._pointer[partition] = (self._pointer[partition] + batch_size) \
                                       % self._question[partition].shape[0]
            logger.debug('next batch at pointer: %d'%(self._pointer[partition]))
            return batch_image_feat, batch_question, batch_answer_label, batch_map_label
        else:
            logger.debug('new epoch of data iteration')
            next_pointer = (self._pointer[partition] + batch_size) \
                           % self._question[partition].shape[0]
            batch_question = self._question[partition][self._pointer[partition]:]
            batch_question = np.append(batch_question,
                                       self._question[partition][:next_pointer],
                                       axis = 0)
            batch_image_id = self._image_id[partition][self._pointer[partition]:]
            batch_image_id = np.append(batch_image_id,
                                       self._image_id[partition][:next_pointer],
                                       axis = 0)

            # index - 1 as query for image feature
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()

            batch_answer = self._answer[partition][self._pointer[partition]:]
            batch_answer = np.append(batch_answer,
                                     self._answer[partition][:next_pointer],
                                     axis = 0)
            batch_answer_label = [self.rng.choice(ans)for ans in batch_answer]
            batch_answer_label = np.array(batch_answer_label)
            batch_map_label = self._map_label[partition][self._pointer[partition]:]
            batch_map_label = np.append(batch_map_label,
                                           self._map_label[partition][:next_pointer],
                                           axis = 0)

            self._pointer[partition] = next_pointer
            logger.debug('next batch at pointer: %d'%(next_pointer))
            return batch_image_feat, batch_question, batch_answer_label, batch_map_label
