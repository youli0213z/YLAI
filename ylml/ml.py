import json
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from matplotlib_inline import backend_inline
torch.manual_seed(0)
def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset.

    Defined in :numref:`sec_fashion_mnist`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

#线性标准化函数
def LinearNormalization(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))
#展平函数
def Flatten(x):
    batch_size = x.shape[0]
    return x.reshape(batch_size,-1)

#线性回归模型
class LinearRegression(nn.Module):
    def __init__(self,input_num,output_num,bias = True):
        super(LinearRegression,self).__init__()
        self.model = nn.Linear(input_num,output_num)
    def forward(self,x):
        return self.model(x)

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
    def start_train(self,trainloader,validloader = None,val_idx = None):
        self.trainloader = trainloader
        self.validloader = validloader
        self.loss_train_list = []
        self.loss_valid_list = []
        self.accurary_rate_train = []
        self.accurary_rate_valid = []
        if val_idx != None:
            self.max_valid_num = int(self.max_epochs / val_idx)
            self.val_idx = val_idx
        if self.task_type == 'REG':
            if isinstance(self.model,nn.Module):
                self.model.train()
            print('Start Train!')
            for epoch in range(self.max_epochs):
                for idx,(x,t) in enumerate (self.trainloader):
                    #if self.model.device = 'cpu'
                    t_hat = self.model(x.to(self.device))
                    loss_ = self.loss_function(t_hat,t.to(self.device))
                    self.optimizer.zero_grad()
                    loss_.backward(retain_graph=True)
                    self.optimizer.step()
                    loss = loss_.item()
                    self.loss_train_list.append(loss)
                print('Train_set Epoch [{}/{}] loss: {}'.format(epoch, self.max_epochs, loss))
        if self.task_type == 'Multi_CLS':
            if isinstance(self.model, nn.Module):
                self.model.train()
            for epoch in range(self.max_epochs):
                total_num = 0
                accurary_num = 0
                for idx,(x,t) in enumerate (self.trainloader):
                    #if self.model.device = 'cpu'
                    x = Flatten(x)
                    total_num += x.shape[0]
                    t_hat = self.model(x.to(self.device))
                    loss_ = self.loss_function(t_hat, t.to(self.device))
                    accurary_num += sum(torch.argmax(t_hat,dim = 1) == t.to(self.device))
                    loss_ = self.loss_function(t_hat,t.to(self.device))
                    self.optimizer.zero_grad()
                    loss_.backward(retain_graph=True)
                    self.optimizer.step()
                    loss = loss_.item()
                    # if idx+1 % self.val_idx == 0:
                self.loss_train_list.append(loss)
                accurary_rate = round(accurary_num.cpu().item()/total_num,4)
                self.accurary_rate_train.append(accurary_rate)
                print('Train_set Epoch [{}/{}] loss: {}, acc: {}'.format(epoch, self.max_epochs, loss, accurary_rate))
                if (epoch+1) % self.val_idx == 0:
                    valid_num = int((epoch+1) / self.val_idx)
                    if isinstance(self.model, nn.Module):
                        self.model.eval()
                    with torch.no_grad():
                        total_num = 0
                        accurary_num = 0
                        for idx, (x, t) in enumerate(self.validloader):
                            # if self.model.device = 'cpu'
                            x = Flatten(x)
                            total_num += x.shape[0]
                            t_hat = self.model(x.to(self.device))
                            loss_ = self.loss_function(t_hat, t.to(self.device))
                            accurary_num += sum(torch.argmax(t_hat, dim=1) == t.to(self.device))
                            loss_ = self.loss_function(t_hat, t.to(self.device))
                            loss = loss_.item()
                            # if idx+1 % self.val_idx == 0:
                        self.loss_valid_list.append(loss)
                        accurary_rate = round(accurary_num.cpu().item() / total_num, 4)
                        self.accurary_rate_valid.append(accurary_rate)
                        print('Start Validation!')
                        print('Valid_set Epoch [{}/{}] loss: {}, acc: {}'.format(valid_num, self.max_valid_num, loss, accurary_rate))
                        print('Stop Validation!')

    def show_loss_value(self):
        n_train_loss_value = len(self.loss_train_list)
        loss_train_list_ = np.array(self.loss_train_list)
        loss_min ,loss_max = np.min(loss_train_list_),np.max(loss_train_list_)
        set_figsize(figsize=(4, 3))
        plt.plot(list(range(n_train_loss_value)), self.loss_train_list, 'b-', linewidth=1, label='Train_loss')
        plt.title('loss_curve')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.ylim(loss_min, loss_max)
        plt.show()
    def show_loss_acc_value(self):
        n_train_loss_value = len(self.loss_train_list)
        n_accurary_rate_train = len(self.accurary_rate_train)
        set_figsize(figsize=(4, 3))
        plt.plot(list(range(n_accurary_rate_train)),self.accurary_rate_train,'r-',linewidth = 1,label = 'Train_acc')
        plt.plot(list(range(n_train_loss_value)), self.loss_train_list, 'b-', linewidth=1, label='Train_loss')
        if self.loss_valid_list != []:
            n_valid_loss_value = len(self.loss_valid_list)
            n_accurary_rate_valid = len(self.accurary_rate_valid)
            plt.plot(list(range(n_accurary_rate_valid)), self.accurary_rate_valid, 'y-', linewidth=1, label='Valid_acc')
            plt.plot(list(range(n_valid_loss_value)), self.loss_valid_list, 'g-', linewidth=1, label='Valid_loss')
        plt.title('loss_curve')
        plt.xlabel('Epochs')
        plt.ylabel('loss_acc')
        plt.legend()
        plt.ylim(0, 1)
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
        self.y = self.y.unsqueeze(1,)
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
        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.scatter(self.x.data.numpy(), self.y.data.numpy())
        plt.title('LinearRegDataset')
        plt.subplot(222)
        plt.scatter(self.x.data.numpy(), self.y.data.numpy())
        plt.title('LinearRegResult')
        plt.plot(x, y, 'r.', markersize=2)
        plt.show()



