#!/usr/bin/env python

import sys
dataDir = '/home/s1670404/VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
sys.path.insert(0, '%s/PythonEvaluationTools/' %(dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import json
import random
import os

# set up file names and paths
taskType    ='OpenEnded'
dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType ='val22014'
annFile     ='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    ='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
# imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

result_file = sys.argv[1]

resFile = '%s/Results/%s'%(dataDir, result_file)

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
