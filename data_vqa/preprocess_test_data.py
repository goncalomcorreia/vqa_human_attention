#/usr/bin/env Python

import sys
dataDir = '/afs/inf.ed.ac.uk/group/synproc/Goncalo/VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
sys.path.insert(0, '%s/PythonEvaluationTools/vqaEvaluation' %(dataDir))

import numpy as np
import pickle as pkl

from vqa import VQA
from process_function import *

with open('/afs/inf.ed.ac.uk/group/synproc/Goncalo/data_vqa/test_imids.pkl', 'r') as f:
    image_ids = pkl.load(f)

# dictionary
with open('/afs/inf.ed.ac.uk/group/synproc/Goncalo/data_vqa/question_dict.pkl', 'r') as f:
    question_key = pkl.load(f)

############################
# processing test questions #
############################
taskType    = 'OpenEnded'
dataType    = 'mscoco'
dataSubType = 'test2015'
annFile     = '%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    = '%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir      ='%s/Images/%s/' %(dataDir, dataSubType)

vqa = VQA(annFile, quesFile)
test_question_ids = vqa.getQuesIds()
test_image_ids = vqa.getImgIds()
test_question_idx = []
test_answer_idx = []
test_answer_counter = []
idx_to_remove = []

for idx, q_id in enumerate(test_question_ids):

    question = vqa.loadQuestion(q_id)
    question = process_sentence(question[0])
    question = question.split()
    question_idx = [question_key[word] if word in question_key
                    else question_key['<unk>'] for word in question ]
    test_question_idx.append(question_idx)
    if idx % 1000 == 0:
        print 'finished processing %d in test'%(idx)

# transform image ids to idx
test_image_ids = [image_ids.index(id) for id in test_image_ids]

test_question_ids = np.array(test_question_ids)
test_image_ids = np.array(test_image_ids)
test_question_idx = np.array(test_question_idx)


# #####################
# # # dumping to disk #
# #####################

with open('test.pkl', 'w') as f:
    pkl.dump(test_question_ids, f)
    pkl.dump(test_image_ids, f)
    pkl.dump(test_question_idx, f)

print 'finished dumping data'
