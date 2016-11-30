#/usr/bin/env Python

import pdb
import sys
dataDir = '/home/zichaoy/VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
sys.path.insert(0, '%s/PythonEvaluationTools/vqaEvaluation' %(dataDir))

import os
import re
from collections import Counter  #
import numpy as np
import random
import h5py
import pickle as pkl

from vqa import VQA
from process_function import *

# k = int(float(sys.argv[1]))
k = 1000


# processing image features #


# process the image list and feature
train_feat_file = 'train2014_fixed_fc7_feat.pkl'
val_feat_file = 'val2014_fixed_fc7_feat.pkl'
with open(train_feat_file) as f:
    train_image_names = pkl.load(f)
    train_image_ids = pkl.load(f)
    train_image_feature = pkl.load(f)

with open(val_feat_file) as f:
    val_image_names = pkl.load(f)
    val_image_ids = pkl.load(f)
    val_image_feature = pkl.load(f)

# combine train and val
image_ids = train_image_ids + val_image_ids
image_ids = [int(float(i)) for i in image_ids]
image_feat = np.zeros((len(image_ids), 4096), dtype='float32')
image_feat[0:len(train_image_ids)] = train_image_feature
image_feat[len(train_image_ids):] = val_image_feature

print 'finished processing features'

################################
# # process the train question #
################################
taskType    = 'OpenEnded'
dataType    = 'mscoco'
dataSubType = 'train2014'
annFile     = '%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    = '%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir      ='%s/Images/%s/' %(dataDir, dataSubType)

vqa = VQA(annFile, quesFile)
train_question_ids = vqa.getQuesIds()
train_image_ids = vqa.getImgIds()
train_questions = []
train_answers = []
question_dict_count = dict()
answer_dict_count = dict()

for idx, q_id in enumerate(train_question_ids):
    question = vqa.loadQuestion(q_id)[0]
    question = process_sentence(question)
    question = question.split()
    for word in question:
        question_dict_count[word] = question_dict_count.get(word, 0) + 1
    answer = vqa.loadAnswer(q_id)[0]
    answer_new = [process_answer(ans) for ans in answer]
    for word in answer_new:
        answer_dict_count[word] = answer_dict_count.get(word, 0) + 1
    train_questions.append(question)
    train_answers.append(answer)
    if idx % 1000 == 0:
        print 'finished processing %d in train' %(idx)


# transform image ids to idx
train_image_ids = [image_ids.index(id) for id in train_image_ids]

# sort question dict
question_count = question_dict_count.values()
sorted_index = [count[0] for count in
                sorted(enumerate(question_count), key = lambda x : x[1],
                       reverse=True)]
sorted_count = sorted(question_count, reverse=True)
question_key = question_dict_count.keys()
question_key = [question_key[idx] for idx in sorted_index]
# add '<unk>' to the begining
question_key.insert(0, '<unk>')
# '<unk>' begins at 1, 0 is reserved for empty words
question_key = dict((key, idx + 1) for idx, key in enumerate(question_key))


# sort answer dict and get top k answers
del answer_dict_count['']
answer_count = answer_dict_count.values()  #
sorted_index = [count[0] for count in
                sorted(enumerate(answer_count), key = lambda x : x[1],
                       reverse=True)]
sorted_count = sorted(answer_count, reverse=True)
answer_key = answer_dict_count.keys()
answer_key = [answer_key[idx] for idx in sorted_index]
answer_top_k = answer_key[:k]
answer_top_k = dict((key, idx) for idx, key in enumerate(answer_top_k))


# convert words to idx and remove some
train_question_idx = []
train_answer_idx = []
train_answer_counter = []
idx_to_remove = []
for idx, answer in enumerate(train_answers):
    question_idx = [question_key[word] for word in train_questions[idx]]
    train_question_idx.append(question_idx)
    answer_idx = [answer_top_k[ans] for ans in answer
                 if ans in answer_top_k]
    answer_counter = Counter(answer_idx)
    train_answer_counter.append(answer_counter)
    train_answer_idx.append(answer_idx)
    if not answer_idx:
        idx_to_remove.append(idx)
print '%d out of %d, %f of the question in train are removed'\
    %(len(idx_to_remove), len(train_question_ids),
      len(idx_to_remove) / float(len(train_question_ids)))

# transform to array and delete all the empty answer
train_question_ids = np.array(train_question_ids)
train_image_ids = np.array(train_image_ids)
train_question_idx = np.array(train_question_idx)
train_answer_idx = np.array(train_answer_idx)
train_answer_counter = np.array(train_answer_counter)

train_question_ids = np.delete(train_question_ids, idx_to_remove)
train_image_ids = np.delete(train_image_ids, idx_to_remove)
train_question_idx = np.delete(train_question_idx, idx_to_remove)
train_answer_idx = np.delete(train_answer_idx, idx_to_remove)
train_answer_counter = np.delete(train_answer_counter, idx_to_remove)

# reshuffle the train data
idx_shuffle = range(train_question_ids.shape[0])
random.shuffle(idx_shuffle)
train_question_ids = train_question_ids[idx_shuffle]
train_image_ids = train_image_ids[idx_shuffle]
train_question_idx = train_question_idx[idx_shuffle]
train_answer_idx = train_answer_idx[idx_shuffle]
train_answer_counter = train_answer_counter[idx_shuffle]
# the most frequent as label
train_answer_label = [counter.most_common(1)[0][0]
                      for counter in train_answer_counter]
train_answer_label = np.array(train_answer_label)

# transform from counter to dict
train_answer_counter = [dict(counter) for counter in train_answer_counter]
train_answer_counter = np.array(train_answer_counter)

print 'finished processing train'

############################
# processing val questions #
############################
taskType    = 'OpenEnded'
dataType    = 'mscoco'
dataSubType = 'val2014'
annFile     = '%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    = '%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir      ='%s/Images/%s/' %(dataDir, dataSubType)

vqa = VQA(annFile, quesFile)
val_question_ids = vqa.getQuesIds()
val_image_ids = vqa.getImgIds()
val_question_idx = []
val_answer_idx = []
val_answer_counter = []
idx_to_remove = []

for idx, q_id in enumerate(val_question_ids):
    answer = vqa.loadAnswer(q_id)[0]
    answer = [process_answer(ans) for ans in answer]
    answer_idx = [answer_top_k[ans] for ans in answer
                  if ans in answer_top_k]
    # none of the answer appear in the top k list
    if not answer_idx:
        idx_to_remove.append(idx)
        # add a false label
        answer_idx = [20000]
    answer_counter = Counter(answer_idx)
    question = vqa.loadQuestion(q_id)
    question = process_sentence(question[0])
    question = question.split()
    question_idx = [question_key[word] if word in question_key
                    else question_key['<unk>'] for word in question ]
    val_question_idx.append(question_idx)
    val_answer_idx.append(answer_idx)
    val_answer_counter.append(answer_counter)
    if idx % 1000 == 0:
        print 'finished processing %d in val'%(idx)

print '%d out of %d, %f of the question in train are removed'\
    %(len(idx_to_remove), len(val_question_ids),
      len(idx_to_remove) / float(len(val_question_ids)))

# transform image ids to idx
val_image_ids = [image_ids.index(id) for id in val_image_ids]

val_question_ids = np.array(val_question_ids)
val_image_ids = np.array(val_image_ids)
val_question_idx = np.array(val_question_idx)
val_answer_idx = np.array(val_answer_idx)
val_answer_label = [counter.most_common(1)[0][0]
                    for counter in val_answer_counter]
val_answer_label = np.array(val_answer_label)
# transform from counter to dict
val_answer_counter = [dict(counter) for counter in val_answer_counter]
val_answer_counter = np.array(val_answer_counter)

val_empty_question_ids = val_question_ids[idx_to_remove]
val_empty_image_ids = val_image_ids[idx_to_remove]
val_empty_question_idx = val_question_idx[idx_to_remove]
val_empty_answer_idx = val_answer_idx[idx_to_remove]
val_empty_answer_counter = val_answer_counter[idx_to_remove]
val_empty_answer_label = val_answer_label[idx_to_remove]

val_question_ids = np.delete(val_question_ids, idx_to_remove)
val_image_ids = np.delete(val_image_ids, idx_to_remove)
val_question_idx = np.delete(val_question_idx, idx_to_remove)
val_answer_idx = np.delete(val_answer_idx, idx_to_remove)
val_answer_counter = np.delete(val_answer_counter, idx_to_remove)
val_answer_label = np.delete(val_answer_label, idx_to_remove)

# split the val to val1 and val2

print 'finished processing val'
val1_idx = random.sample(range(val_question_ids.shape[0]),
                         val_question_ids.shape[0]/2)
val2_idx = range(val_question_ids.shape[0])
val2_idx = [ i for i in val2_idx if i not in val1_idx ]

val1_question_ids = val_question_ids[val1_idx]
val1_image_ids = val_image_ids[val1_idx]
val1_question_idx = val_question_idx[val1_idx]
val1_answer_idx = val_answer_idx[val1_idx]
val1_answer_label = val_answer_label[val1_idx]
val1_answer_counter = val_answer_counter[val1_idx]

val2_question_ids = val_question_ids[val2_idx]
val2_image_ids = val_image_ids[val2_idx]
val2_question_idx = val_question_idx[val2_idx]
val2_answer_idx = val_answer_idx[val2_idx]
val2_answer_label = val_answer_label[val2_idx]
val2_answer_counter = val_answer_counter[val2_idx]


val2_empty_idx = random.sample(range(val_empty_question_ids.shape[0]),
                               val_empty_question_ids.shape[0]/2)
val2_all_question_ids = np.concatenate([val2_question_ids,
                                        val_empty_question_ids[val2_empty_idx]],
                                        axis=0)
val2_all_image_ids = np.concatenate([val2_image_ids,
                                     val_empty_image_ids[val2_empty_idx]],
                                    axis=0)
val2_all_question_idx = np.concatenate([val2_question_idx,
                                        val_empty_question_idx[val2_empty_idx]],
                                       axis=0)
val2_all_answer_idx = np.concatenate([val2_answer_idx,
                                      val_empty_answer_idx[val2_empty_idx]],
                                     axis=0)
val2_all_answer_label = np.concatenate([val2_answer_label,
                                        val_empty_answer_label[val2_empty_idx]],
                                       axis=0)
val2_all_answer_counter = np.concatenate([val2_answer_counter,
                                          val_empty_answer_counter[val2_empty_idx]],
                                         axis=0)


# #####################
# # # dumping to disk #
# #####################

# image feature
image_feat_h5 = h5py.File('image_feat_256_fixed.h5', 'w')
image_feat_h5['feat'] = image_feat
image_feat_h5.close()

with open('image_list.pkl', 'w') as f:
    pkl.dump(np.array(image_ids), f)

# dictionary
with open('question_dict.pkl', 'w') as f:
    pkl.dump(question_key, f)

with open('answer_dict.pkl', 'w') as f:
    pkl.dump(answer_top_k, f)

# train
with open('train.pkl', 'w') as f:
    pkl.dump(train_question_ids, f)
    pkl.dump(train_image_ids, f)
    pkl.dump(train_question_idx, f)
    pkl.dump(train_answer_idx, f)
    pkl.dump(train_answer_counter, f)
    pkl.dump(train_answer_label, f)

# val
with open('val1.pkl', 'w') as f:
    pkl.dump(val1_question_ids, f)
    pkl.dump(val1_image_ids, f)
    pkl.dump(val1_question_idx, f)
    pkl.dump(val1_answer_idx, f)
    pkl.dump(val1_answer_counter, f)
    pkl.dump(val1_answer_label, f)

with open('val2.pkl', 'w') as f:
    pkl.dump(val2_question_ids, f)
    pkl.dump(val2_image_ids, f)
    pkl.dump(val2_question_idx, f)
    pkl.dump(val2_answer_idx, f)
    pkl.dump(val2_answer_counter, f)
    pkl.dump(val2_answer_label, f)

with open('val2_all.pkl', 'w') as f:
    pkl.dump(val2_all_question_ids, f)
    pkl.dump(val2_all_image_ids, f)
    pkl.dump(val2_all_question_idx, f)
    pkl.dump(val2_all_answer_idx, f)
    pkl.dump(val2_all_answer_counter, f)
    pkl.dump(val2_all_answer_label, f)

print 'finished dumping data'
