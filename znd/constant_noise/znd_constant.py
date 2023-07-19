import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


I = 0.3
I2 = 1
class ZNDConstant(Optimizer):
    
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        self.I = I
        self.I2 = I2
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,weight_decay=weight_decay,nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ZNDConstant, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ND, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    def step(self, epoch, closure=None):
        grad_dict = {}  # initial grad_dict
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            count = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                grad_dict[epoch] = torch.clone(d_p).detach()  # store current gradient
                if weight_decay != 0:
                    # gradient = gradient + weight+decay
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.clone(d_p).detach()  # initial I
                    else:
                        I_buf = param_state['I_buffer']
                        prev_grad = grad_dict.get(epoch-1)
                        if prev_grad is None:
                            prev_grad = torch.clone(d_p).detach()
                        I_buf.mul_(momentum).add_(prev_grad, alpha=1 - dampening)
                    if count == 0:
                        d_p.add_(0.01)
                    count += 1      
                    if epoch < 100:    
                        d_p.add_(I_buf,alpha = self.I)
                    else:
                        d_p.add_(I_buf,alpha = self.I2)
                
                
                p.add_(d_p, alpha=-group['lr'])    

                
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'grad_buffer' not in param_state:
                #         g_buf = param_state['grad_buffer'] = torch.zeros_like(p.data)
                #         g_buf = d_p
                #         I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                #         I_buf.mul_(momentum).add_(g_buf)  # I_buf = I_buf * momentum + g_buf
                #     else:
                #         I_buf = param_state['I_buffer']
                #         g_buf = param_state['grad_buffer'] # get the previous step gradient
                #         I_buf.mul_(momentum).add_(g_buf, alpha=1 - dampening)  # I_buf = I_buf * momentum + (1 - dampenin) * g_buf
                #         self.state[p]["grad_buffer"]=d_p.clone() #store gradient for next iteration
                   
                #     d_p = d_p.add_(I_buf, alpha=self.I)
                    
                # p.data.add_(d_p, alpha=-group['lr'])  # p.data = p.data + lr * (d_p + I * I_buf + D * D_buf)

        return loss
