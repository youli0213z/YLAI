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
    def __init__(self,input_num,output_num,bias = True):
        super(Linear, self).__init__()
        self.input_num =input_num
        self.output_num = output_num
        self.bias = bias
        self.params = None
    def parameters(self):
        params_dict = {}
        w = torch.randn(self.input_num,self.output_num,requires_grad=True)
        self.w = w
        w = nn.Parameter(torch.randn(self.input_num,self.output_num,requires_grad=True))
        params_dict['w']= w
        if self.bias == True:
            b = torch.zeros(self.input_num,self.output_num,requires_grad=True)
            self.b = b
            b = nn.Parameter(torch.zeros(self.input_num,self.output_num,requires_grad=True))
            params_dict['b']= b
            self.b = b
        else:
            self.b = torch
        self.params = params_dict
        return params_dict.values()
    def forward(self,x):
        x = torch.matmul(x,self.w.T)

        return


