import torch
class SGD:
    def __init__(self, params, batch_size, lr=0.01, momentum=0, weight_decay=0, dampening=0, nesterov=False):
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
                param -= self.lr * d_p / self.batch_size


class MB_SGD:
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0, dampening=0, nesterov=False):
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