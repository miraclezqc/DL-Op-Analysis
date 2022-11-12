import csv
import json
import os
import ast
import numpy as np
import torch
from sample_config import *
from judge import *


def get_sample_config():
    with open("./mse_loss.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["x1"][i], arg_data["x2"][i], 
            arg_data["reduction"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        url="",  # noqa
        tags=[],
    )

def gen_np_args(x1_, x2_, reduction_):
    x1 = np.random.random(x1_).astype(np.float32)
    x2 = np.random.random(x2_).astype(np.float32)

    return [x1, x2, reduction_]



def args_adaptor(np_args):
    x1 = torch.from_numpy(np_args[0]).cuda()
    x2 = torch.from_numpy(np_args[1]).cuda()
    reduction_ = 'mean'
    if np_args[2] != "":
        if not argIsNone(np_args[2]):
            if isinstance(np_args[2], list):
                reduction_ = np_args[2][0]
            else:
                reduction_ = np_args[2]
    return [x1, x2, reduction_]



def get_input_data():
    with open("./mse_loss.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    in_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["input"][i]:
            in_size *= dim
        in_sizes.append(in_size)
    return in_sizes
    