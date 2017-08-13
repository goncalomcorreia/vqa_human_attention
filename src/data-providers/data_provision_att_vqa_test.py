#!/usr/bin/env python

from data_provision_att_vqa import *
import time

logger = logging.getLogger('root')

class DataProvisionAttVqaTest(DataProvisionAttVqa):
    def __init__(self, data_folder, feature_file, rng = None, state = None, n_shuffles = None):
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
        self._splits = ['test']
        self._pointer = OrderedDict()
        if rng is None:
            rng = np.random.RandomState(1234)
        self.rng = rng
        for split in self._splits:
            with open(os.path.join(data_folder, split) + '_v2.pkl') as f:
                split_question_id = pkl.load(f)
                split_image_id = pkl.load(f)
                split_question = pkl.load(f)
            idx = range(split_question.shape[0])
            self._question_id[split] = split_question_id[idx]
            self._image_id[split] = split_image_id[idx]
            self._question[split] = split_question[idx]
            self._answer[split] = np.zeros(split_question.shape[0])
            self._answer_counter[split] = np.zeros(split_question.shape[0])
            self._answer_label[split] = np.zeros(split_question.shape[0])
            self._pointer[split] = 0

        if n_shuffles is not None:
            for i in xrange(n_shuffles):
                self.random_shuffle()

        if state is not None:
            self.rng.set_state(state)

        logger.info('finished loading data')
