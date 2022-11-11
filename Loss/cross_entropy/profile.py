import argparse
import torch
from torch.nn import functional
import numpy as np
import time
from sample import *

def cross_entropy(input, target, weight_,
                size_average_, ignore_index_, reduce_, reduction_, label_smoothing_):
    output = functional.cross_entropy(input, target, weight=weight_,
            size_average=size_average_, ignore_index=ignore_index_, 
            reduce=reduce_, reduction=reduction_, label_smoothing=label_smoothing_)
    return output


def profile(device):   
    samples = get_sample_config()
    print(samples._args_cases[0])
    frist_args = samples._args_cases[0]
    np_arg = gen_np_args(frist_args[0], frist_args[1], frist_args[2], frist_args[3],
                         frist_args[4], frist_args[5], frist_args[6], frist_args[7])
    torch_arg = args_adaptor(np_arg)
    print(torch_arg)
    out = cross_entropy(torch_arg[0], torch_arg[1], torch_arg[2], torch_arg[3],
                  torch_arg[4], torch_arg[5], torch_arg[6], torch_arg[7])
    print(out)


def main():
    # cuda settings
    use_cuda = torch.cuda.is_available()
    assert use_cuda == True, "cuda environment is not ready"
    device = torch.device("cuda")
    profile(device)

if __name__ == '__main__':
    main()