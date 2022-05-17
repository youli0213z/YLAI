import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import numpy
import time

#机器学习算法
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(3,1)
    def forward(self,x):
        return self.model(x)
class LinearRegression_:
    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.params = [self.w,self.b]
    def parameters(self):
        return self.params
    def forward(self,x):
        return torch.matmul(x,self.w) + self.b
    def __call__(self, x):
        return self.forward(x)

#优化器
class Optim:
    def __int__(self):
        pass
    '''
    小批量随机梯度下降法（Stochastic Gradient Descent）
    '''
    class SGD:
        def __init__(self, params,batch_size, lr=0.01, momentum=0, weight_decay=0, dampening=0, nesterov=False):
            self.params = params
            self.batch_size = batch_size
            self.lr = lr
            self.weight_decay = weight_decay
            self.momentum = momentum
            self.dampening = dampening
            self.nesterov = nesterov

        def zero_grad(self):

            for param in self.params:
                if param.grad == None:
                    continue
                else:
                    param.grad.zero_()

        @torch.no_grad()
        def step(self):
            for param in self.params:
                if param.grad == None:
                    continue
                else:
                    d_p = param.grad.data
                    if self.weight_decay != 0:  # 进行正则化
                        # add_表示原处改变，d_p = d_p + weight_decay*p.data
                        d_p.add_(self.weight_decay, param.data)
                    if self.momentum != 0:
                        param_state = self.state[param]  # 之前的累计的数据，v(t-1)
                        # 进行动量累计计算
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            # 之前的动量
                            buf = param_state['momentum_buffer']
                            # buf= buf*momentum + （1-dampening）*d_p
                            buf.mul_(self.momentum).add_(1 - self.dampening, d_p)
                        if self.nesterov:  # 使用neterov动量
                            # d_p= d_p + momentum*buf
                            d_p = d_p.add(self.momentum, buf)
                        else:
                            d_p = buf
                        # p = p - lr*d_p
                    param -= self.lr * d_p /self.batch_size
    class MB_SGD:
        def __init__(self,params,lr = 0.01,momentum = 0,weight_decay = 0,dampening=0,nesterov=False):
            self.params = params
            self.lr = lr
            self.weight_decay = weight_decay
            self.momentum = momentum
            self.dampening = dampening
            self.nesterov = nesterov
        def zero_grad(self):

            for param in self.params:
                if param.grad == None:
                    continue
                else:
                    param.grad.zero_()

        @torch.no_grad()
        def step(self):
            for param in self.params:
                if param.grad == None:
                    continue
                else:
                    d_p = param.grad.data
                    if self.weight_decay != 0:  # 进行正则化
                        # add_表示原处改变，d_p = d_p + weight_decay*p.data
                        d_p.add_(self.weight_decay, param.data)
                    if self.momentum != 0:
                        param_state = self.state[param]  # 之前的累计的数据，v(t-1)
                        # 进行动量累计计算
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            # 之前的动量
                            buf = param_state['momentum_buffer']
                            # buf= buf*momentum + （1-dampening）*d_p
                            buf.mul_(self.momentum).add_(1 - self.dampening, d_p)
                        if self.nesterov:  # 使用neterov动量
                            # d_p= d_p + momentum*buf
                            d_p = d_p.add(self.momentum, buf)
                        else:
                            d_p = buf
                        # p = p - lr*d_p
                    param -= self.lr * d_p

    # @staticmethod
    #def SGD(params,batch_size,lr = 0.01):
        # with torch.no_grad():
        #     for param in params:
        #         if param.grad == None:
        #             pass
        #         else:
        #             param -= lr * param.grad / batch_size
        #             param.grad.zero_()

    # @staticmethod
    # def MB_SGD(params, lr=0.01):
    #     with torch.no_grad():
    #         for param in params:
    #             if param.grad == None:
    #                 pass
    #             else:
    #                 param -= lr * param.grad
    #                 param.grad.zero_()
#损失函数
class LossFunction:
    def __init__(self):
        pass

    @staticmethod
    def MSE(x,y,device = 'cpu'):
        if torch.is_tensor(x) == False:
            x = torch.tensor(x, device=device)
        if torch.is_tensor(y) == False:
            y = torch.tensor(y, device=device)

        return torch.matmul((x-y).T,(x-y))/x.shape[0] #(x-y.reshape(x.shape)) ** 2 / 2torch.matmul((x-y).T,(x-y))/x.shape[0]

    @staticmethod
    def BCE(x,y,device='cpu'):
        if torch.is_tensor(x) == False:
            x = torch.tensor(x, device=device)
        if torch.is_tensor(y) == False:
            y = torch.tensor(y, device=device)
        l = 0
        for i in range(x.shape[0]):
            l -= torch.log(x[i])*y[i] - torch.log(1-x[i])*(1-y[i])
        return l

    @staticmethod
    def CrossEntropy(x, y, class_num, device='cpu'):
        if torch.is_tensor(x) == False:
            x = torch.tensor(x, device=device)
        if torch.is_tensor(y) == False:
            y = torch.tensor(y, device=device)
        l = 0
        for i in range(x.shape[0]):
            l -= torch.log(x[i][y[i]])
        return l
#激活函数
class ActivationFunction:
    def __init__(self):
        pass

    @staticmethod
    def Sigmoid(x,device = 'cpu'):
        if torch.is_tensor(x) == False:
            x = torch.tensor(x, device=device)
        return 1 / (1 + torch.exp(-1 * x))

    @staticmethod
    def show_Sigmoid():
        x = torch.tensor(np.linspace(-10, 10, 1000))
        y = 1 / (1 + torch.exp(-1 * x))
        plt.figure()
        plt.plot(x, y, 'r.', markersize=2)
        plt.xticks((-10, 10))
        plt.title('sigmoid')
        plt.show()

    @staticmethod
    def Tanh(x,device = 'cpu'):
        if torch.is_tensor(x) == False:
            x = torch.tensor(x, device=device)
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    @staticmethod
    def show_Tanh():
        x = torch.tensor(np.linspace(-10, 10, 1000))
        y = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        plt.figure()
        plt.plot(x, y, 'r.', markersize=3)
        plt.xticks((-10, 10))
        plt.title('Tanh')
        plt.show()

    @staticmethod
    def Relu(x,device = 'cpu'):
        if torch.is_tensor(x) == False:
            x = torch.tensor(x,device = device)
        n,p = x.shape
        x_ = torch.ones([n,p],device = device)
        for i in range(n):
            for j in range(p):
                x_[i,j] = torch.where(x[i,j]>=0,x[i,j],0)
        return x_

    @staticmethod
    def show_Relu():
        x = np.linspace(-10, 10, 1000)
        y = [i if i >= 0 else 0 for i in x]
        plt.figure()
        plt.plot(x, y, 'r.', markersize=2)
        plt.xticks((-10, 10))
        plt.title('Relu')
        plt.show()

    @staticmethod
    def Softmax(x,device = 'cpu'):
        if torch.is_tensor(x) == False:
            x = torch.tensor(x,device = device)
        return torch.exp(x) / torch.sum(torch.exp(x),axis = 1,keepdim=True)
#神经网络模块
class nn:
    def __init__(self):
        pass
    class Linear:
        def __init__(self,w,bias = True,device = 'cpu'):
            self.w = w
            if torch.is_tensor(w) == False:
                self.w[:,0] = torch.tensor(w[:,0]*bias, device = device)
            else:
                self.w[:,0] = w[:,0] * bias

            self.device = device
        def forward(self,x,):
            self.x = torch.tensor(x, device = self.device)
            print(self.x.type(),self.w.type())
            x = torch.matmul(self.x,self.w)
            return x
        def backward(self,dout):
            dx = torch.mm(dout,self.w.T)
            dw = torch.mm(self.x.T,dout)
            return dw
    class MSE:
        def __init__(self,regulariztion=''):
            self.regulariztion = regulariztion
            self.loss = None
            self.y = None
            self.t =None

        def forward(self,y,t):
            self.y = y
            self.t = t
            self.loss = LossFunction.MSE(self.y,self.t)
            #if self.regulariztion == 'L1':
                #self.loss += 1/2 * torch.mm(self.w,self.w.T)
            return self.loss

        def backward(self,dout = 1):
            batch_size = self.t.shape[0]
            dx = 2/batch_size * (self.y-self.t)*dout
            #dw1 = self.w * dout
            return dx
#训练器
class Train_:
    def __init__(self,max_epochs,lr,loss_function,optimizer,model,task_type = '',device ='cpu'):
        self.max_epochs = max_epochs
        self.lr = lr
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model
        self.task_type = task_type
    def start_train1(self,dataloader,val_idx,params,batch_size):
        self.dataloader = dataloader
        self.val_idx = val_idx
        self.params = params
        self.batch_size = batch_size
        self.loss_list = []
        self.optimizer = self.optimizer(params,self.batch_size,self.lr)
        if self.task_type == 'REG':
            for epoch in range(self.max_epochs):
                for idx,(x,t) in enumerate (self.dataloader):
                    loss_ = self.loss_function(self.model(x.to(self.device),*self.params),t.to(self.device))
                    self.optimizer.zero_grad()
                    loss_.backward()
                    self.optimizer.step()
                    loss = loss_.item()
                    if idx == self.val_idx - 1:
                        self.loss_list.append(loss)
                        print('{}epoch {}idx时损失值为{}'.format(epoch,idx,loss))
        return self.params
    def start_train2(self,dataloader,val_idx,params):
        self.dataloader = dataloader
        self.val_idx = val_idx
        self.loss_list = []
        self.model = self.model.to(self.device)
        self.params = params
        self.optimizer = self.optimizer(params,self.lr)
        if self.task_type == 'REG':
            for epoch in range(self.max_epochs):
                for idx,(x,t) in enumerate (self.dataloader):
                    loss_ = self.loss_function(self.model(x.to(self.device)),t.to(self.device))
                    self.optimizer.zero_grad()
                    loss_.backward()
                    self.optimizer.step()
                    loss = loss_.item()
                    if idx == self.val_idx - 1:
                        self.loss_list.append(loss)
                        print('{}epoch {}idx时损失值为{}'.format(epoch,idx,loss))
    def show_loss_value(self):
        n_loss_value = len(self.loss_list)
        plt.figure()
        plt.plot(list(range(n_loss_value)),self.loss_list,'r.')
        plt.show()
class Train:
    def __init__(self,max_epochs,loss_function,optimizer,model,task_type = '',device ='cpu'):
        self.max_epochs = max_epochs
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model
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
                    loss_ = self.loss_function(self.model(x.to(self.device)),t.to(self.device))
                    self.optimizer.zero_grad()
                    loss_.backward()
                    self.optimizer.step()
                    loss = loss_.item()
                    if idx == self.val_idx - 1:
                        self.loss_list.append(loss)
                        print('{}epoch {}idx时损失值为{}'.format(epoch,idx,loss))
        return self.params
    def show_loss_value(self):
        n_loss_value = len(self.loss_list)
        plt.figure()
        plt.plot(list(range(n_loss_value)),self.loss_list,'r.')
        plt.show()














