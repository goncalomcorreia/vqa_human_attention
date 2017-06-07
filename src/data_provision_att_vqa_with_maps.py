#!/usr/bin/env python

from data_provision_att_vqa import *

logger = logging.getLogger('root')

class DataProvisionAttVqaWithMaps(DataProvisionAttVqa):
    def __init__(self, data_folder, feature_file, maps_data_folder):
        super(DataProvisionAttVqaWithMaps, self).__init__(
            data_folder, feature_file
        )
        self._att_maps = OrderedDict()
        self._att_maps_qids = OrderedDict()
        for maps_split in ['train', 'val']:
            with open(os.path.join(maps_data_folder, maps_split+'_att_maps') + '.pkl') as f:
                self._att_maps[maps_split] = pkl.load(f)
                self._att_maps_qids[maps_split] = pkl.load(f)
                self._att_maps_qids[maps_split] = [int(elem) for elem in self._att_maps_qids[maps_split]]
        aux_att_maps = []
        self._att_maps_qids['val'] = list(set(self._att_maps_qids['val']))
        for i, elem in enumerate(self._att_maps['val']):
            if i%3==0:
                aux_att_maps.append(elem)
        self._att_maps['val'] = np.array(aux_att_maps)
        self._qid_map_info = OrderedDict()
        for split in self._splits[:-1]:
            maps_split = split
            if split!='train':
                maps_split = 'val'
            map_qids_bool = [True if elem in self._att_maps_qids[maps_split] else False for elem in self._question_id[split]]
            self._qid_map_info[split] = {elem1: elem2 for elem1, elem2 in zip(self._question_id[split], map_qids_bool)}
        self._qid_map_info['trainval1'] = dict(self._qid_map_info['train'].items() + self._qid_map_info['val1'].items())
        self._map_label = OrderedDict()
        for split in self._splits:
            map_idx = [list(self._question_id[split]).index(elem)
                                              for elem in self._question_id[split]
                                              if self._qid_map_info[split][elem]]
            self._question_id[split] = self._question_id[split][map_idx]
            self._image_id[split] = self._image_id[split][map_idx]
            self._question[split] = self._question[split][map_idx]
            self._answer[split] = self._answer[split][map_idx]
            self._answer_counter[split] = self._answer_counter[split][map_idx]
            self._answer_label[split] = self._answer_label[split][map_idx]
            self._map_label[split] = np.array([self.get_map_from_qid(elem) for elem in self._question_id[split]])

    def has_map(self, qid):
        for split in self._splits:
            if qid in self._qid_map_info[split].keys():
                return self._qid_map_info[split][qid]
        return False

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
            batch_answer_label = [random.choice(ans)for ans in batch_answer]
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
            batch_answer_label = [random.choice(ans)for ans in batch_answer]
            batch_answer_label = np.array(batch_answer_label)
            batch_map_label = np.append(batch_map_label,
                                           self._map_label[partition][:next_pointer],
                                           axis = 0)

            self._pointer[partition] = next_pointer
            logger.debug('next batch at pointer: %d'%(next_pointer))
            return batch_image_feat, batch_question, batch_answer_label, batch_map_label
