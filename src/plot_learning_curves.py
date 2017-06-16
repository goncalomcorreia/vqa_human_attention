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

for array_path in arrays_path:
    with np.load(array_path) as data:
        err_valid.append(data['valid_error'])
        acc_valid.append(data['valid_accuracy'])
        err_map_valid.append(data['valid_error_map'])

fig = plt.figure(figsize=(5, 7))

ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)


for i in xrange(len(arrays_path)):
    ax1.plot(err_valid[i], label=legend[i])
    ax2.plot(acc_valid[i], label=legend[i])
    ax3.plot(err_map_valid[i], label=legend[i])

ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Validation set cross entropy')
ax1.set_yscale('log')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation Accuracy')
ax3.legend(loc=0)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Validation set map cross entropy')
ax3.set_yscale('log')
fig.savefig(name, bbox_inches='tight')
