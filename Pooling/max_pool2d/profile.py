import argparse
import torch
from torch.nn import functional
import numpy as np
import time
from sample import *


def max_pool2d(input_torch, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
    print(len(input_torch), kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    output = functional.max_pool2d(input_torch, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    return output


def virtual_arg_profile():
    pow = 28 #2^1 - 2^28
    for i in range(0,pow):
        shape = [2**(i+1)]
        print(shape)
        np_arg = gen_np_args(shape, shape, "mean")
        torch_arg = args_adaptor(np_arg)
        out = max_pool2d(torch_arg[0], torch_arg[1])
        print(out)



def profile(device):   
    samples = get_sample_config()
    samples_num = len(samples._args_cases)
    for i in range(samples_num):
        case_args = samples._args_cases[i]
        np_arg = gen_np_args(case_args[0], case_args[1], case_args[2], case_args[3], case_args[4], case_args[5], case_args[6])
        torch_arg = args_adaptor(np_arg)
        out = max_pool2d(torch_arg[0], torch_arg[1], torch_arg[2], torch_arg[3], torch_arg[4], torch_arg[5], torch_arg[6])


def main():
    # cuda settings
    use_cuda = torch.cuda.is_available()
    # assert use_cuda == True, "cuda environment is not ready"
    device = torch.device("cuda")
    profile(device)
    virtual_arg_profile()

if __name__ == '__main__':
    main()
