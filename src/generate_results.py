#!/usr/bin/env python

import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')
import sys
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/')
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/data-providers/')
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/models/')
from optimization_weight import *
from data_provision_att_vqa import *
from data_processing_vqa import *
import json
import pickle
import sys
dataDir = '/afs/inf.ed.ac.uk/group/synproc/Goncalo/VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
sys.path.insert(0, '%s/PythonEvaluationTools/' %(dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

f = open('/afs/inf.ed.ac.uk/group/synproc/Goncalo/data_vqa/answer_dict.pkl', 'r')
answer_dict = pickle.load(f)
f.close()
answer_dict = {v: k for k, v in answer_dict.iteritems()}

model_path = sys.argv[1]
result_file_name = sys.argv[2]
model_script = sys.argv[3]

result = OrderedDict()

if model_script == 'baseline':
    from san_att_conv_twolayer_theano import *
    options, params, shared_params = load_model(model_path)
    image_feat, input_idx, input_mask, label, \
    dropout, cost, accu, pred_label, \
    prob_attention_1, prob_attention_2 = build_model(
        shared_params, options)
elif model_script=='hsan_deepfix':
    from semi_joint_hsan_deepfix_att_theano import *
    options, params, shared_params = load_model(model_path)
    options['saliency_dropout'] = 0.5
    image_feat, input_idx, input_mask, \
    label, dropout, ans_cost, accu, pred_label, \
    prob_attention_1, prob_attention_2, map_cost, map_label = build_model(shared_params, params, options)
elif model_script=='hsan':
    from semi_joint_hsan_att_theano import *
    options, params, shared_params = load_model(model_path)
    image_feat, input_idx, input_mask, \
    label, dropout, ans_cost, accu, pred_label, \
    prob_attention_1, prob_attention_2, map_cost, map_label = build_model(shared_params, options)
elif model_script=='hsan_deepfix_split':
    from multi_joint_hsan_deepfix_att_theano import *
    options, params, shared_params = load_model(model_path)
    image_feat, input_idx, input_mask, \
    label, dropout, ans_cost, accu, pred_label, \
    prob_attention_1, prob_attention_2, map_cost, \
    map_label, saliency_attention = build_model(shared_params, params, options)


f_pass = theano.function(
    inputs=[
        image_feat,
        input_idx,
        input_mask],
    outputs=[pred_label],
    on_unused_input='warn')

options['data_path'] = '/afs/inf.ed.ac.uk/group/synproc/Goncalo/data_vqa/'

data_provision_att_vqa = DataProvisionAttVqa(options['data_path'], options['feature_file'])

dropout.set_value(numpy.float32(0.))

i=0

for batch_image_feat, batch_question, batch_answer_label in data_provision_att_vqa.iterate_batch(
        options['val_split'], options['batch_size']):
    input_idx, input_mask = process_batch(
        batch_question, reverse=options['reverse'])
    batch_image_feat = reshape_image_feat(batch_image_feat,
                                          options['num_region'],
                                          options['region_dim'])
    [pred_label] = f_pass(
        batch_image_feat,
        np.transpose(input_idx),
        np.transpose(input_mask))
    res =[]
    for pred in pred_label:
        ans = answer_dict[pred]
        ques_id = data_provision_att_vqa._question_id['val2'][i]
        i += 1
        result[ques_id] = ans

results = [result]

d = results[0]
res = []
for key, value in d.iteritems():
    res.append({'answer': value, 'question_id': int(key)})

with open('/afs/inf.ed.ac.uk/group/synproc/Goncalo/VQA/Results/'+result_file_name, 'w') as outfile:
    json.dump(res, outfile)

# set up file names and paths
taskType    ='OpenEnded'
dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType ='val22014'
annFile     ='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    ='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
# imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

resFile = '%s/Results/%s'%(dataDir, result_file_name)

# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=4)   #n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
"""
If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the question ids in annotation file
"""
vqaEval.evaluate()

# print accuracies
print "\n"
print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
print "Per Question Type Accuracy is the following:"
for quesType in vqaEval.accuracy['perQuestionType']:
	print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
print "\n"
print "Per Answer Type Accuracy is the following:"
for ansType in vqaEval.accuracy['perAnswerType']:
	print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
print "\n"
