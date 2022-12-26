import csv
import json
import os
import ast
import numpy as np
import torch


class SampleConfig(object):
    def __init__(self,
                 args_cases=[],
                 requires_grad=[],
                 backward=False,
                 warm_up_iters=10,
                 performance_iters=1000,
                 timeline_iters=10,
                 save_timeline=False,
                 rtol=1e-04,
                 atol=1e-04,
                 url=None,
                 tags=[]):
        assert len(args_cases) > 0
        self._args_cases = args_cases
        self._requires_grad = requires_grad
        self._backward = backward
        self._warm_up_iters = warm_up_iters
        self._performance_iters = performance_iters
        self._timeline_iters = timeline_iters
        self._save_timeline = save_timeline
        self._rtol = rtol
        self._atol = atol
        self._url = url
        self._tags = tags

    @property
    def args_cases(self):
        return self._args_cases

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def backward(self):
        return self._backward

    @property
    def warm_up_iters(self):
        return self._warm_up_iters

    @property
    def performance_iters(self):
        return self._performance_iters

    @property
    def timeline_iters(self):
        return self._timeline_iters

    @property
    def save_timeline(self):
        return self._save_timeline

    @property
    def rtol(self):
        return self._rtol

    @property
    def atol(self):
        return self._atol

    @property
    def tags(self):
        return self._tags

    def show_info(self):
        tags = ""
        for tag in self._tags:
            tags = tags + tag.value + " "
        return self._url, tags

    def show(self):
        url, tags = self.show_info()
        print("url:", url)
        print("tags:", tags)


def argIsNone(args):
    if isinstance(args, list):
        assert len(args) == 1
        if args[0] == None :
            return True
        elif isinstance(args[0], str):
            if args[0].lower() == 'none':
                return True
        else:
            return False
    else:
        if args == None:
            return True
        elif isinstance(args, str):
            if args.lower == "none":
                return True
        else:
            return False


def get_sample_config():
    with open("max_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input_size"][i], arg_data["kernel_size"][i], arg_data["stride"][i], arg_data["padding"][i], arg_data["dilation"][i], arg_data["ceil_mode"][i], arg_data["return_indices"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 7,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        url="",  # noqa
        tags=[],
    )

def gen_np_args(input_size_, kernel_size_, stride_, padding_, dilation_, ceil_mode_, return_indices_):
    input_np = np.random.random(input_size_).astype(np.float32)
    kernel_size = kernel_size_
    stride = stride_
    padding = padding_
    dilation = dilation_
    ceil_mode = ceil_mode_
    return_indices = return_indices_

    return [input_np, kernel_size, stride, padding, dilation, ceil_mode, return_indices]



def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0])
    kernel_size = np_args[1][0]
    stride = np_args[2][0]
    padding = np_args[3][0]
    dilation = np_args[4][0]
    ceil_mode = np_args[5][0]
    return_indices = np_args[6][0]

    return [input_torch, kernel_size, stride, padding, dilation, ceil_mode, return_indices]

# def get_input_data():
#     with open("max_pool2d.json", "r") as f:
#         arg_data = json.load(f)
#     arg_data_length = len(arg_data["input_size"])
#     in_sizes = []
#     for i in range(arg_data_length):
#         in_size = 1
#         for dim in arg_data["input_size"][i]:
#             in_size *= dim
#         in_sizes.append(in_size)
#     return in_sizes

def get_input_output_data():
    with open("max_pool2d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    out_sizes = []
    kernel_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        in_HW_size = 1
        idx = 0
        for dim in arg_data["input_size"][i]:
            in_size *= dim
            if idx == 2 or idx == 3: # HW
                in_HW_size *= dim
            idx += 1
        in_sizes.append(in_size)

        assert len(arg_data["kernel_size"][i]) == 1
        assert len(arg_data["stride"][i]) == 1
        assert len(arg_data["padding"][i]) == 1
        assert len(arg_data["dilation"][i]) == 1

        k = arg_data["kernel_size"][i][0]
        stride = arg_data["stride"][i][0]
        pad = arg_data["padding"][i][0]
        dilation = arg_data["dilation"][i][0]

        out_h = int(np.floor((arg_data["input_size"][i][2] + 2 * pad  - dilation * (k-1) - 1) / stride)) + 1
        out_w = int(np.floor((arg_data["input_size"][i][3] + 2 * pad  - dilation * (k-1) - 1) / stride)) + 1

        out_size = out_h * out_w
        out_sizes.append(out_size * in_size/ in_HW_size)
        kernel_sizes.append(in_HW_size/out_size)
    return in_sizes, out_sizes, kernel_sizes


def get_dedu_input_data():
    with open("max_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    out_sizes = []
    kernel_sizes = []
    for i in range(arg_data_length):
        if arg_data["input_size"][i][0] * arg_data["input_size"][i][1] * arg_data["input_size"][i][2] * arg_data["input_size"][i][3]:
            in_size = 1
            in_HW_size = 1
            idx = 0
            for dim in arg_data["input_size"][i]:
                in_size *= dim
                if idx == 2 or idx == 3: # HW
                    in_HW_size *= dim
                idx += 1

            k = arg_data["kernel_size"][i][0]
            stride = arg_data["stride"][i][0]
            pad = arg_data["padding"][i][0]
            dilation = arg_data["dilation"][i][0]

            out_h = int(np.floor((arg_data["input_size"][i][2] + 2 * pad  - dilation * (k-1) - 1) / stride)) + 1
            out_w = int(np.floor((arg_data["input_size"][i][3] + 2 * pad  - dilation * (k-1) - 1) / stride)) + 1

            out_size = out_h * out_w

            in_sizes.append(in_size)
            out_sizes.append(out_size * in_size/ in_HW_size)
            kernel_sizes.append(in_HW_size/out_size)
    return in_sizes, out_sizes, kernel_sizes

def get_dedu_input_info():
    with open("max_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_infos = []
    out_infos = []
    kernel_infos = []
    for i in range(arg_data_length):
        if arg_data["input_size"][i][0] * arg_data["input_size"][i][1] * arg_data["input_size"][i][2] * arg_data["input_size"][i][3] :
            in_info = str(arg_data["input_size"][i][0])
            for dim in range(len(arg_data["input_size"][i])-1):
                in_info += "x" + str(arg_data["input_size"][i][dim+1])
            # in_info += "  " + str(arg_data["output_size"][i][0]) + "x" + str(arg_data["output_size"][i][1])
            in_infos.append(in_info)

            k = arg_data["kernel_size"][i][0]
            stride = arg_data["stride"][i][0]
            pad = arg_data["padding"][i][0]
            dilation = arg_data["dilation"][i][0]

            out_h = int(np.floor((arg_data["input_size"][i][2] + 2 * pad  - dilation * (k-1) - 1) / stride)) + 1
            out_w = int(np.floor((arg_data["input_size"][i][3] + 2 * pad  - dilation * (k-1) - 1) / stride)) + 1

            out_info =  str(out_h) + "x" + str(out_w) #+ " " + str(arg_data["ceil_mode"][i][0]) + " " +  str(arg_data["return_indices"][i][0])
            # out_info = str(arg_data["input_size"][i][0]) +"x"+ str(arg_data["input_size"][i][1]) + "x" + str(out_h) + "x" + str(out_w)
            out_infos.append(out_info)
            
            # kernel_info = str(arg_data["input_size"][i][2]//arg_data["output_size"][i][0] * arg_data["input_size"][i][3]//arg_data["output_size"][i][1])
            # kernel_info = str(arg_data["input_size"][i][0] * arg_data["input_size"][i][1])
            # kernel_info = str(arg_data["input_size"][i][0] * arg_data["input_size"][i][1] * arg_data["input_size"][i][2] * arg_data["input_size"][i][3] /  arg_data["output_size"][i][0] // arg_data["output_size"][i][1])
            kernel_info = in_info + " " + str(k) + " " + str(k)
            # kernel_info = str(arg_data["output_size"][i][0]) + str(arg_data["output_size"][i][1]) + str(arg_data["input_size"][i][0]) +  str(arg_data["input_size"][i][0]) 
            NC_size =  arg_data["input_size"][i][0] * arg_data["input_size"][i][1]
            # if NC_size > 2048:
            #     kernel_infos.append(kernel_info)
            kernel_infos.append(kernel_info)
    return in_infos,out_infos,kernel_infos

