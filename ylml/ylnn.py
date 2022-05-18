import torch
from torch import nn
from torchvision.datasets import CIFAR10,FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
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

    def forward(self,x,params = None,model_id = None):
        pass

    def __call__(self,x,params = None,model_id = None):
        return self.forward(x)

    def to(self,device = 'cpu'):
        if self.model_name == '':
            if self.params_exist == True:
                with torch.no_grad():
                    self.w = self.w.to(device)
                    self.w.requires_grad = True
                    self.b = self.b.to(device)
                    self.b.requires_grad = True
                    self.params = [self.w,self.b]
        else:
            for model in self.model:
                if model.params_exist == True:
                    with torch.no_grad():
                        model.w = model.w.to(device)
                        model.w.requires_grad = True
                        model.b = model.b.to(device)
                        model.b.requires_grad = True
                        model.params = [model.w, model.b]
                        self.params += model.params
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

    def forward(self, x,params = None,model_id = None):
        if params == None:
            x = torch.matmul(x, self.params[0]) + self.params[1]
        else:
            x = torch.matmul(x, params[model_id]) + params[model_id]
        return x
class Relu(ylModule):
    def __init__(self):
        super(Relu, self).__init__()
        self.layer_name = ['Relu']
        #self.model = [Relu()]
        self.params_exist = False
    def forward(self,x,params = None,model_id =None):
        return Relu_(x)

class Sigmoid(ylModule):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.layer_name = ['Simoid']
        #self.model = [Simoid()]
        self.params_exist = False
    def forward(self,x,params = None,model_id =None):
        return Sigmoid_(x)



class Tanh(ylModule):
    def __init__(self):
        super(Tanh, self).__init__()
        self.layer_name = ['Tanh']
        #self.model = [Tanh()]
        self.params_exist = False
    def forward(self,x,params = None,model_id =None):
        return Tanh_(x)

class LeakyRelu(ylModule):
    def __init__(self):
        super(LeakyRelu, self).__init__()
        self.layer_name = ['LeakyRelu']
        #self.model = [LeakyRelu()]
        self.params_exist = False
    def forward(self,x,params = None,model_id =None):
        return LeakyRelu_(x)

class Softmax(ylModule):
    def __init__(self):
        super(Softmax, self).__init__()
        self.layer_name = ['Softmax']
        #self.model = [LeakyRelu()]
        self.params_exist = False
    def forward(self,x,params = None,model_id =None):
        return Softmax_(x)
class MLP_(ylModule):
    def __init__(self,input_num,hidden_num_list,output_num,activationFunction = 'Relu',task_type ='REG'):
        super(MLP_, self).__init__()
        self.activationFunction_dict = {'Relu':Relu(),'Sigmoid':Sigmoid(),'Tanh':Tanh(),'Softmax':Softmax()}
        self.input_num = input_num
        self.hidden_num_list = hidden_num_list
        self.output_num = output_num
        self.activationFunction = self.activationFunction_dict[activationFunction]
        self.task_type = task_type
        self.model = []
        self.layer_name = []
        self.model_name = 'MLP'
        self.hidden_num = len(hidden_num_list)
        self.model += [Linear(self.input_num,hidden_num_list[0]),self.activationFunction]
        for i in range(0,self.hidden_num-1):
            self.model += [Linear(hidden_num_list[i],hidden_num_list[i+1]),self.activationFunction]
        self.model += [Linear(hidden_num_list[-1],self.output_num)]
        if self.task_type == 'Binary_CLS':
            self.model += [Sigmoid()]
        elif self.task_type == 'Multi_CLS':
            self.model += [Softmax()]
        for model in self.model:
            self.layer_name += model.layer_name
            self.params += model.params
    def forward(self,x,params =True,model_id =0):
        for model in self.model:
            x = model(x,self.params,model_id)
            if model.layer_name[0] not in self.activationFunction_dict.keys():
                model_id += 1
        return x
class MLP(nn.Module):
    def  __init__(self,input_num,hidden_num_list,output_num,activationFunction = 'Relu',task_type ='REG'):
        super(MLP, self).__init__()
        self.activationFunction_dict = {'Relu': nn.ReLU(), 'Sigmoid': nn.Sigmoid(), 'Tanh': nn.Tanh(), 'Softmax': nn.Softmax()}
        self.input_num = input_num
        self.hidden_num_list = hidden_num_list
        self.output_num = output_num
        self.activationFunction = self.activationFunction_dict[activationFunction]
        self.task_type = task_type
        self.model = nn.Sequential()
        self.hidden_num = len(hidden_num_list)
        self.model_idx = 0
        self.model.add_module(str(self.model_idx),nn.Linear(self.input_num, hidden_num_list[0]))
        self.model[self.model_idx].weight.data.normal_(0,0.01)
        self.model[self.model_idx].bias.data.fill_(0)
        self.model_idx += 1
        self.model.add_module(str(self.model_idx), self.activationFunction)
        self.model_idx += 1
        for i in range(0, self.hidden_num - 1):
            self.model.add_module(str(self.model_idx), nn.Linear(hidden_num_list[i], hidden_num_list[i + 1]))
            self.model[self.model_idx].weight.data.normal_(0, 0.01)
            self.model[self.model_idx].bias.data.fill_(0)
            self.model_idx += 1
            self.model.add_module(str(self.model_idx), self.activationFunction)
            self.model_idx += 1
        self.model.add_module(str(self.model_idx),nn.Linear(hidden_num_list[-1], self.output_num))
        self.model[self.model_idx].weight.data.normal_(0, 0.01)
        self.model[self.model_idx].bias.data.fill_(0)
        self.model_idx += 1
        # if self.task_type == 'Binary_CLS':
        #     self.model.add_module(str(self.model_idx),nn.Sigmoid())
        #     self.model_idx += 1
        # elif self.task_type == 'Multi_CLS':
        #     self.model.add_module(str(self.model_idx),nn.Softmax(dim=1))
        #     self.model_idx += 1
    def forward(self, x):
        x = self.model(x)
        return x

#class Sequential()
class CIFAR10_MLP():
    def __init__(self,root ='./',train = True,transform = transforms.ToTensor(),download = True,batch_size = 4):
        self.CIFAR10_MLP_train_dataset = CIFAR10(root = root,train = train,transform = transform,download = download)
        self.CIFAR10_MLP_valid_dataset = CIFAR10(root = root,train = False,transform = transform,download = download)
        self.batch_size = batch_size
        self.CIFAR10_MLP_train_dataloader = self.get_dataloader_(self.CIFAR10_MLP_train_dataset)
        self.CIFAR10_MLP_valid_dataloader = self.get_dataloader_(self.CIFAR10_MLP_valid_dataset)

    def get_dataloader_(self,CIFAR10_MLP_dataset):
        CIFAR10_MLP_dataloader = DataLoader(CIFAR10_MLP_dataset,batch_size = self.batch_size,shuffle=True)
        return CIFAR10_MLP_dataloader

    def get_dataloader(self):
        return self.CIFAR10_MLP_train_dataloader,self.CIFAR10_MLP_valid_dataloader

class FashionMNIST_MLP():
    def __init__(self,root ='./',train = True,transform = transforms.ToTensor(),download = True,batch_size = 4):
        self.FashionMNIST_MLP_train_dataset = FashionMNIST(root = root,train = train,transform = transform,download = download)
        self.FashionMNIST_MLP_valid_dataset = FashionMNIST(root = root,train = False,transform = transform,download = download)
        self.batch_size = batch_size
        self.FashionMNIST_MLP_train_dataloader = self.get_dataloader_(self.FashionMNIST_MLP_train_dataset)
        self.FashionMNIST_MLP_valid_dataloader = self.get_dataloader_(self.FashionMNIST_MLP_valid_dataset)

    def get_dataloader_(self,FashionMNIST_MLP_dataset):
        FashionMNIST_MLP_dataloader = DataLoader(FashionMNIST_MLP_dataset,batch_size = self.batch_size,shuffle=True)
        return FashionMNIST_MLP_dataloader

    def get_dataloader(self):
        return self.FashionMNIST_MLP_train_dataloader,self.FashionMNIST_MLP_valid_dataloader







