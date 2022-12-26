import csv
import json
import os
import numpy as np
from sample import *


column = 40
in_sizes,out_sizes,kernel_sizes = get_input_output_data()



all_size = len(in_sizes)
print(np.sum(kernel_sizes)/all_size)

print(all_size)

input_mem_data = np.array([(i * 4.0 /1024/ 1024 ) for i in in_sizes])  # KB

# input_mem_data = np.array([(i) for i in kernel_sizes])  

print(np.max(input_mem_data))
print(np.min(input_mem_data))
step = int(np.ceil((np.max(input_mem_data) - np.min(input_mem_data)) / column))
# step = int(np.ceil((1200 - np.min(input_mem_data)) / column))
# step = int(np.ceil((5000 - np.min(input_mem_data)) / column))

mem_dict = {}
label_data = []

# split the range
start = int(np.min(input_mem_data))
for col in range(column):
    end = (int)(start + step)
    label = str(start) + '-' + str(end)
    mem_dict[label] = 0
    label_data.append(label)
    start = end
print(mem_dict)

# count the range data
cnt = 0
for in_data in input_mem_data:
    if True:
    # if in_data <= 1200:
    # if in_data <= 5000:
        cnt += 1
        mem_dict[label_data[int(np.floor(in_data/step))]] += 1
print("cnt = ", cnt)
print(mem_dict)
precent = np.array([value / all_size * 100 for value in mem_dict.values()]) 
print(precent)
print(np.min(input_mem_data))
print(np.max(input_mem_data))

# -------------------------

import matplotlib.pyplot as plt

fontsize_tune = 7
title_size = 10
colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y']

allrown = 1
fig, axes = plt.subplots(nrows=allrown, ncols=1, figsize=(11,8*allrown))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.8)    #subplots创建多个子图


########################## the first row ################################
rown = 0
ax0 = axes

axis = ax0
axis.set_ylabel('Ratio(%)', fontsize=10)
# data = list(mem_dict.values())
data = precent
names = list(mem_dict.keys())

axis.bar(names, data, color=colors)
axis.set_title('Memory size distribution of input data(MB)'
               , fontsize=title_size)
# axis.set_title('Memory size distribution of output data under 1.2MB(KB)'
#                , fontsize=title_size)
# axis.set_title('Ratio distribution of input to output size under 5000'
            #    , fontsize=title_size)
axis.tick_params(axis='x', labelsize=fontsize_tune)
axis.tick_params(axis='y', labelsize=fontsize_tune)
axis.set_xticks(range(0,len(names),1))
axis.set_xticklabels(names,rotation=45)


plt.savefig('adaptive_avg_pool2d.png', dpi=600)