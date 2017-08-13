#/usr/bin/env Python

import sys
dataDir = '/Users/goncalocorreia/VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
sys.path.insert(0, '%s/PythonEvaluationTools/vqaEvaluation' %(dataDir))

import numpy as np
import pickle as pkl

from vqa import VQA
from process_function import *
import json

with open('/Users/goncalocorreia/vqa_human_attention/data_vqa/test_imids.pkl', 'r') as f:
    image_ids = pkl.load(f)

image_ids = [int(image_id) for image_id in image_ids]

# dictionary
with open('/Users/goncalocorreia/vqa_human_attention/data_vqa/question_dict.pkl', 'r') as f:
    question_key = pkl.load(f)

############################
# processing test questions #
############################
version     = 'v2'
taskType    = 'OpenEnded'
dataType    = 'mscoco'
dataSubType = 'test2015'
annFile     = '%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    = '%s/Questions/%s_%s_%s_%s_questions.json'%(dataDir, version, taskType, dataType, dataSubType)
imgDir      ='%s/Images/%s/' %(dataDir, dataSubType)

questions = json.load(open(quesFile, 'r'))
test_question_ids = [x['question_id'] for x in questions['questions']]
test_image_ids = [x['image_id'] for x in questions['questions']]
test_question_idx = []
qa =  {ques['question_id']: ques['question'] for ques in questions['questions']}

for idx, q_id in enumerate(test_question_ids):

    question = qa[q_id]
    question = process_sentence(question)
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

with open('/Users/goncalocorreia/vqa_human_attention/data_vqa/test_v2.pkl', 'w') as f:
    pkl.dump(test_question_ids, f)
    pkl.dump(test_image_ids, f)
    pkl.dump(test_question_idx, f)

print 'finished dumping data'
