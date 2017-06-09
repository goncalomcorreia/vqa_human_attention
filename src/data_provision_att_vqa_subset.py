#!/usr/bin/env python

from data_provision_att_vqa import *
import time

logger = logging.getLogger('root')

class DataProvisionAttVqaSubset(DataProvisionAttVqa):
    def __init__(self, data_folder, feature_file, maps_data_folder):
        super(DataProvisionAttVqaSubset, self).__init__(
            data_folder, feature_file
        )
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
        for split in self._splits:
            map_idx = np.where(np.in1d(self._question_id[split], qids_with_maps))[0]
            self._question_id[split] = self._question_id[split][map_idx]
            self._image_id[split] = self._image_id[split][map_idx]
            self._question[split] = self._question[split][map_idx]
            self._answer[split] = self._answer[split][map_idx]
            self._answer_counter[split] = self._answer_counter[split][map_idx]
            self._answer_label[split] = self._answer_label[split][map_idx]
