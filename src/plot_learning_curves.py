#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

name = sys.argv[1]
arrays_path = sys.argv[2]

with np.load(arrays_path) as data:
    err_train = data['train_error']
    err_valid = data['train_accuracy']
    acc_train = data['valid_error']
    acc_valid = data['valid_accuracy']

fig = plt.figure(figsize=(12, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.plot(err_train, label='Training Error')
ax2.plot(err_valid, label='Validation Error')
ax3.plot(acc_train, label='Training Accuracy')
ax4.plot(acc_valid, label='Validation Accuracy')

ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set error')
ax1.set_yscale('log')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation set error')
ax2.set_yscale('log')
ax3.legend(loc=0)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training Accuracy')
ax4.legend(loc=0)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation Accuracy')
fig.savefig(name, bbox_inches='tight')
