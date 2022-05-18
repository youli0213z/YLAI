import json
import torch
import numpy as np
from ylml import ylnn
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
torch.manual_seed(0)
#线性回归模型
class LinearRegression(nn.Module):
    def __init__(self,input_num,output_num,bias = True):
        super(LinearRegression,self).__init__()
        self.model = nn.Linear(input_num,output_num)
    def forward(self,x):
        return self.model(x)

class LinearRegression_(ylnn.ylModule):
    def __init__(self, input_num, output_num, bias=True):
        super(LinearRegression_, self).__init__()
        self.input_num = input_num
        self.output_num = output_num
        self.w = torch.randn(self.input_num, self.output_num, requires_grad=True)
        self.idx = 0
        self.layer_name = 'LinearRegression_'
        self.layer_name_idx = self.layer_name + str(self.idx)
        if bias == True:
            self.b = torch.zeros(self.input_num, self.output_num, requires_grad=True)
        else:
            self.b = torch.zeros_like(self.w)
        self.params = [self.w,self.b]

    def parameters(self):
        return self.params

    def forward(self, x):
        x = torch.matmul(x, self.params[0].T) + self.params[1]
        return x
    def get_weight(self):
        weight_dict = {self.layer_name_idx:self.params}
        return weight_dict
    def get_weight_json(self):
        self.layer_name_idx = self.layer_name + str(self.idx)
        params = [param.cpu().tolist() for param in self.params]
        weight_dict = {self.layer_name_idx: params}
        with open(self.layer_name_idx+'.json','w') as weight_json:
            weight_json_ = json.dump(weight_dict,weight_json)
    def load_weight_json(self,weight_json_file_path,device ='cpu'):
        with open(weight_json_file_path, "r") as weight_json_file:
            weight_dict = json.load(weight_json_file)
        self.w = torch.tensor(weight_dict[self.layer_name_idx][0],requires_grad=True,device = device)
        self.b = torch.tensor(weight_dict[self.layer_name_idx][1], requires_grad=True,device = device)
        self.params = [self.w,self.b]

    # def to(self, device='cpu'):
    #     for param in self.params:
    #         param = param.to(device)
    #     return self
#训练器
class Train:
    def __init__(self,max_epochs,loss_function,optimizer,model,task_type = '',device ='cpu'):
        self.max_epochs = max_epochs
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model.to(device)
        self.task_type = task_type
        self.params = None
    def start_train(self,dataloader,val_idx):
        self.dataloader = dataloader
        self.val_idx = val_idx
        self.loss_list = []
        if self.task_type == 'REG':
            for epoch in range(self.max_epochs):
                for idx,(x,t) in enumerate (self.dataloader):
                    #if self.model.device = 'cpu'
                    t_hat = self.model(x.to(self.device))
                    loss_ = self.loss_function(t_hat,t.to(self.device))
                    self.optimizer.zero_grad()
                    loss_.backward()
                    self.optimizer.step()
                    loss = loss_.item()
                    if idx == self.val_idx - 1:
                        self.loss_list.append(loss)
                        print('{}epoch {}idx时损失值为{}'.format(epoch,idx+1,loss))
        return self.params
    def show_loss_value(self):
        n_loss_value = len(self.loss_list)
        plt.figure()
        plt.plot(list(range(n_loss_value)),self.loss_list,'r.')
        plt.title('loss_curve')
        plt.show()
#线性回归数据集
class LinearRegDataset_(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx, ::], self.y[idx, ::]

    def __len__(self):
        return self.x.shape[0]

class LinearRegDataset:
    def __init__(self,sample_num,x_dim,w,b,batch_size,device = 'cpu'):
        self.x = torch.randn(sample_num,x_dim,requires_grad=True,device = device,dtype=torch.float32)
        self.y = self.x[:,0] * w +  b + torch.randn(sample_num)
        self.y = torch.tensor(self.y,requires_grad=True,device = device).unsqueeze(1,)
        self.batch_size = batch_size

    class LinearRegDataset_(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __getitem__(self, idx):
            return self.x[idx, ::], self.y[idx, ::]

        def __len__(self):
            return self.x.shape[0]
    def get_dataloader(self):
        Dataset = LinearRegDataset_(self.x,self.y)
        Dataloader = DataLoader(Dataset,batch_size = self.batch_size,shuffle = True)
        return Dataloader
    def show_dataset(self):
        plt.figure()
        plt.scatter(self.x.data.numpy(),self.y.data.numpy())
        plt.title('LinearRegDataset')
        plt.show()
    def show_result(self,weight_list):
        x = torch.tensor(np.linspace(-3, 3, 1000),dtype=torch.float32).unsqueeze(1)
        y = torch.matmul(x,weight_list[0].cpu().data.T) + weight_list[1].cpu().data
        plt.figure()
        plt.subplot(221)
        plt.scatter(self.x.data.numpy(), self.y.data.numpy())
        plt.title('LinearRegDataset')
        plt.subplot(222)
        plt.scatter(self.x.data.numpy(), self.y.data.numpy())
        plt.title('LinearRegResult')
        plt.plot(x, y, 'r.', markersize=2)
        plt.show()



