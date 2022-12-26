import argparse
import torch
from torch.nn import functional
import numpy as np
import time
from sample import *


def adaptive_avg_pool2d(input_torch, output_size):
    output = functional.adaptive_avg_pool2d(input_torch, output_size)
    return output



def profile(device):   
    samples = get_sample_config()
    samples_num = len(samples._args_cases)
    for i in range(samples_num):
        case_args = samples._args_cases[i]
        np_arg = gen_np_args(case_args[0], case_args[1])
        torch_arg = args_adaptor(np_arg)
        out = adaptive_avg_pool2d(torch_arg[0], torch_arg[1])


def main():
    # cuda settings
    use_cuda = torch.cuda.is_available()
    # assert use_cuda == True, "cuda environment is not ready"
    device = torch.device("cuda")
    profile(device)

if __name__ == '__main__':
    main()
