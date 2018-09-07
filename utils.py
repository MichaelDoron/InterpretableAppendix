from __future__ import division
import numpy as np
import torch
import json
import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for (param_name,param), (shared_param_name, shared_param) in zip(model.named_parameters(),
                                   shared_model.named_parameters()):
        if (param.requires_grad):
            if shared_param.grad is not None and not gpu:
                return
            elif not gpu:
                shared_param._grad = param.grad
            else:
                try:
                    shared_param._grad = param.grad.cpu()
                except:
                    print(param_name)
                    print(param.requires_grad)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class Merger(torch.nn.Module):
    def __init__(self, in_features):
        super(Merger, self).__init__()
        self.in_features = in_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(torch.zeros(self.in_features))

    def forward(self, input_1, input_2):
        output_1 = torch.nn.functional.linear(input = input_1, weight = torch.diag(self.weight), bias=None)
        output_2 = torch.nn.functional.linear(input = input_2, weight = torch.diag(1 - self.weight), bias=None)
        return output_1 + output_2


class WeightClipper(object):
    def __init__(self):
        pass
    
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight') and isinstance(module, Merger):
            w = module.weight.data
            w = w.clamp(0,1)

