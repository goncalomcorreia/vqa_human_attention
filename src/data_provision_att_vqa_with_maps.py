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
