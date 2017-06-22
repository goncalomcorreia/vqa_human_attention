import sys
dataDir = '../../VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import json
import random
import os

# set up file names and paths
taskType    ='OpenEnded'
dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType ='val2014'
annFile     ='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    ='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
resultType  ='fake'
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']

resFile = '%s/Results/mtl.json'%(dataDir,)

def preprocess(dict_list):
    d = dict_list[0]
    res = []
    for key, value in d.iteritems():
        res.append({'answer': value, 'question_id': int(key)})
    with open('%s/Results/complete_mtl.json'%(dataDir,), 'w') as outfile:
        json.dump(res, outfile)
    return res

def create_truncated_ans_ques(resFile, annFile, quesFile, dataDir):
    val_questions = json.load(open(quesFile))
    val_answers = json.load(open(annFile))
    val2_pred = json.load(open(resFile))

    if len(val2_pred)==1:
        val2_pred = preprocess(val2_pred)

    val2_questions = val_questions.copy()
    val2_answers = val_answers.copy()

    val2_ids = [elem['question_id'] for elem in val2_pred]

    val2_questions['questions'] = []
    for elem in val_questions['questions']:
        if elem['question_id'] in val2_ids:
            val2_questions['questions'].append(elem)

    val2_answers['annotations'] = []
    for elem in val_answers['annotations']:
        if elem['question_id'] in val2_ids:
            val2_answers['annotations'].append(elem)

    with open('%s/Questions/OpenEnded_mscoco_val22014_questions.json'%(dataDir,), 'w') as outfile:
        json.dump(val2_questions, outfile)

    with open('%s/Annotations/mscoco_val22014_annotations.json'%(dataDir,), 'w') as outfile:
        json.dump(val2_answers, outfile)


create_truncated_ans_ques(resFile, annFile, quesFile, dataDir)

# set up file names and paths
taskType    ='OpenEnded'
dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType ='val22014'
annFile     ='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    ='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']

val2_pred = json.load(open(resFile))
val2_pred = preprocess(val2_pred)
resFile = '%s/Results/complete_mtl.json'%(dataDir,)
# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

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
