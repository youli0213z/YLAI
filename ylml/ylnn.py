import torch
from torch import nn
import numpy as np

class ylModule:
    def __init__(self):
        self.params = None
        self.w = None
        self.b = None
        self.params = [self.w,self.b]
    def parameters(self):
        return self.params
    def forward(self,x):
        pass
    def __call__(self, x):
        return self.forward(x)

    def to(self,device = 'cpu'):
        # if self.w is not None:
        #      self.w = torch.clone(self.w).to(device)
            # self.w.require_grad = False
            # self.w.require_grad = True
        # if self.b is not None:
        #      self.b = torch.clone(self.b).to(device)
            # self.b.require_grad = False
            # self.b.require_grad = True
        # self.params = [self.w,self.b]
        with torch.no_grad():
            self.w = self.w.to(device)
            self.w.requires_grad = True
            self.b = self.b.to(device)
            self.b.requires_grad = True
            self.params = [self.w,self.b]
            # self.params = [param.to(device) for param in self.params]
            # self.w = self.params[0]
            # self.b = self.params[1]
        return self

class Linear(ylModule):
    def __init__(self, input_num, output_num, bias=True):
        super(Linear, self).__init__()
        self.input_num = input_num
        self.output_num = output_num
        self.w = torch.randn(self.input_num, self.output_num, requires_grad=True)
        self.idx = 0
        self.layer_name = 'Linear'
        if bias == True:
            self.b = torch.zeros(self.input_num, self.output_num, requires_grad=True)
        else:
            self.b = torch.zeros_like(self.w)
        self.params = [self.w, self.b]

    def parameters(self):
        return self.params

    def forward(self, x):
        x = torch.matmul(x, self.params[0].T) + self.params[1]
        return x


