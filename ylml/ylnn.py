import torch
from torch import nn
import numpy as np
import json
from ylml.ActivationFunction import *
class ylModule:
    def __init__(self):
        self.params_exist = True
        self.model =[]
        self.w = None
        self.b = None
        self.params = []
        self.layer_name = []
        self.model_name = ''
    def parameters(self):
        return self.params
    def forward(self,x):
        pass
    def __call__(self, x):
        return self.forward(x)

    def to(self,device = 'cpu'):
        with torch.no_grad():
            self.w = self.w.to(device)
            self.w.requires_grad = True
            self.b = self.b.to(device)
            self.b.requires_grad = True
            self.params = [self.w,self.b]
        return self
    def get_weight(self):
        weight_dict = {}
        if self.model != []:
            for model in self.model:
                if model.params_exist == False:
                    continue
                else:
                    if weight_dict.get(model.layer_name[0]) == None :
                        weight_dict[model.layer_name[0]] = []
                        params = [param.cpu().tolist() for param in model.params]
                        weight_dict[model.layer_name[0]].append(params)
                    else:
                        params = [param.cpu().tolist() for param in model.params]
                        weight_dict[model.layer_name[0]].append(model.params)
        params = [param.cpu().tolist() for param in self.params]
        weight_dict = {self.layer_name[0]:params}
        return weight_dict
    def get_weight_json(self):
        weight_dict = self.get_weight()
        if self.model_name != '':
            with open(self.model_name + '.json', 'w') as weight_json:
                weight_json_ = json.dump(weight_dict, weight_json)
        else:
            with open(self.layer_name[0] + '.json', 'w') as weight_json:
                weight_json_ = json.dump(weight_dict, weight_json)

    def load_weight_json(self,weight_json_file_path,device='cpu'):
        with open(weight_json_file_path, "r") as weight_json_file:
            weight_dict = json.load(weight_json_file)
        if self.model_name == '':
            self.params = torch.tensor(weight_dict[self.layer_name[0]], requires_grad=True, device=device)
        else:
            for layer in weight_dict.keys():
                len_layer = len(weight_dict[layer])
                while (len_layer>0):
                    for model in self.model:
                        if layer == model.layer_name[0]:
                            model.params = torch.tensor(weight_dict[layer][len_layer-1], requires_grad=True, device=device)
                            model.w = model.params[0]
                            model.b = model.params[1]
                            len_layer -=1
            self.params = [model.params for model in self.model]

class Linear(ylModule):
    def __init__(self, input_num, output_num, bias=True):
        super(Linear, self).__init__()
        self.input_num = input_num
        self.output_num = output_num
        self.params_exist = True
        self.w = torch.randn(self.input_num, self.output_num, requires_grad=True)
        self.layer_name = ['Linear']
        #self.model = [Linear(self.input_num,self.output_num)]
        if bias == True:
            self.b = torch.zeros(1, requires_grad=True)
        else:
            self.b = torch.zeros(1,requires_grad=False)
        self.params = [self.w, self.b]

    def parameters(self):
        return self.params

    def forward(self, x):
        x = torch.matmul(x, self.params[0].T) + self.params[1]
        return x
class Relu(ylModule):
    def __init__(self):
        super(Relu, self).__init__()
        self.layer_name = ['Relu']
        #self.model = [Relu()]
        self.params_exist = False
    def forward(self,x):
        return Relu(x)

class Simoid(ylModule):
    def __init__(self):
        super(Simoid, self).__init__()
        self.layer_name = ['Simoid']
        #self.model = [Simoid()]
        self.params_exist = False
    def forward(self,x):
        return Simoid(x)

class Tanh(ylModule):
    def __init__(self):
        super(Tanh, self).__init__()
        self.layer_name = ['Tanh']
        #self.model = [Tanh()]
        self.params_exist = False
    def forward(self,x):
        return Tanh(x)

class LeakyRelu(ylModule):
    def __init__(self):
        super(LeakyRelu, self).__init__()
        self.layer_name = ['LeakyRelu']
        #self.model = [LeakyRelu()]
        self.params_exist = False
    def forward(self,x):
        return LeakyRelu(x)

class MLP(ylModule):
    def __init__(self,input_num,hidden_num_list,output_num,activationFunction = 'Relu'):
        super(MLP, self).__init__()
        self.input_num = input_num
        self.hidden_num_list = hidden_num_list
        self.output_num = output_num
        self.activationFunction = activationFunction
        self.model = []
        self.layer_name = []
        self.model_name = 'MLP'
        self.hidden_num = len(hidden_num_list)
        self.model += [Linear(self.input_num,hidden_num_list[0]),Relu()]
        for i in range(1,self.hidden_num-1):
            self.model += [Linear(hidden_num_list[i],hidden_num_list[i+1]),Relu()]
        self.model += [Linear(hidden_num_list[-1],self.output_num)]
        for model in self.model:
            self.layer_name += model.layer_name
            self.params += model.params
    def forward(self,x):
        for model in self.model:
            x = model(x)
        return x
class Sequential()




