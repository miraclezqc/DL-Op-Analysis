import argparse
import torch
from torch.nn import functional
import numpy as np
import time
from sample import *


def mse_loss(x1, x2, reduction_):
    output = functional.mse_loss(x1, x2, reduction=reduction_)
    return output


def profile(device):   
    samples = get_sample_config()
    print(samples._args_cases[0])
    frist_args = samples._args_cases[0]
    np_arg = gen_np_args(frist_args[0], frist_args[1], frist_args[2])
    torch_arg = args_adaptor(np_arg)
    print(torch_arg)
    out = mse_loss(torch_arg[0], torch_arg[1], torch_arg[2])
    print(out)


def main():
    # cuda settings
    use_cuda = torch.cuda.is_available()
    assert use_cuda == True, "cuda environment is not ready"
    device = torch.device("cuda")
    profile(device)

if __name__ == '__main__':
    main()
