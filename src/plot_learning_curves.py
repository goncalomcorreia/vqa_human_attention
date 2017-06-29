#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

name = sys.argv[1]
i = 2
arrays_path = []
legend = []
while i<len(sys.argv):
    arrays_path.append(sys.argv[i])
    i += 1
    legend.append(sys.argv[i])
    i += 1

acc_valid = []
err_valid = []
err_map_valid = []
x_axis_valid = []
x_axis_main_train = []
x_axis_sub_train = []
acc_train = []
err_train = []
err_map_train = []


for array_path in arrays_path:
    with np.load(array_path) as data:
        err_valid.append(data['valid_error'])
        acc_valid.append(data['valid_accuracy'])
        x_axis_valid.append(data['x_axis_epochs'])
        #TODO: do this only if it exists
        err_map_valid.append(data['valid_error_map'])
        # err_map_train.append(data['train_error_map'])
        # x_axis_sub_train.append(data['sub_x_axis'])
        # x_axis_main_train.append(data['main_x_axis'])
        # err_train.append(data['train_error'])
        # acc_train.append(data['train_accuracy'])

fig = plt.figure(figsize=(5, 7))

ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)


for i in xrange(len(arrays_path)):
    ax1.plot(x_axis_valid[i], err_valid[i], label="Val "+legend[i])
    #ax1.plot(x_axis_main_train[i], err_train[i], label='Train '+legend[i])
    ax2.plot(x_axis_valid[i], acc_valid[i], label="Val "+legend[i])
    #ax2.plot(x_axis_main_train[i], acc_train[i], label='Train '+legend[i])
    ax3.plot(x_axis_valid[i], err_map_valid[i], label="Val "+legend[i])
    #ax3.plot(x_axis_sub_train[i], err_map_train[i], label='Train '+legend[i])

ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Validation set error')
ax1.set_yscale('log')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation Accuracy')
ax3.legend(loc=0)
ax3.set_xlabel('Epoch number')
#ax3.set_ylabel('Validation set map error')
#ax3.set_yscale('log')
fig.savefig(name, bbox_inches='tight')
