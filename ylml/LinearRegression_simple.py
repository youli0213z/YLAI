import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from ylml import Train,Optim
#构建线性回归数据集
x = torch.randint(-100,100,[500,3],dtype =torch.float32,requires_grad=True )
print(x.shape)
y = x[:,0] * 3 + x[:,1] * -1 + x[:,2] * 20
y = torch.randn(500,requires_grad=True)
y = y.unsqueeze(1,)
print(y.shape)
x_train , x_val = x[0:400,::] , x[400:,::]
y_train , y_val = y[0:400,::] , y[400:,::]
class Dataset_(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __getitem__(self, idx):
        return self.x[idx,::],self.y[idx,::]
    def __len__(self):
        return self.x.shape[0]
train_set = Dataset_(x_train,y_train)
val_set = Dataset_(x_val,y_val)
train_dataloader = DataLoader(train_set,batch_size = 128,shuffle = True)
val_dataloader = DataLoader(train_set,batch_size = 128,shuffle = True)
#定义损失函数
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.model = nn.Sequential(nn.Linear(3,1))
    def forward(self,x):
        return self.model(x)
device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
linearRegression = LinearRegression()
linearRegression.model[0].weight.data.normal_(0,1)
linearRegression.model[0].bias.data.fill_(0)
loss_function  = nn.MSELoss()
optimizer1 = torch.optim.SGD
optimizer2 = Optim.MB_SGD
train = Train(100,0.0001,loss_function,optimizer2,linearRegression,task_type = 'REG',device = device)
train.start_train2(train_dataloader,3,linearRegression.parameters())
train.show_loss_value()
# print(linearRegression.model[0].weight)
