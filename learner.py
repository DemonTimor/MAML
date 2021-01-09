import  torch
from    torch import nn
from    torch.nn import functional as F

class Learner(nn.Module):
    """
    定义一个网络
    """
    def __init__(self, config):
        super(Learner, self).__init__()
        self.config = config ## 对模型各个超参数的定义
        '''
        ## ParameterList可以像普通Python列表一样进行索引，
        但是它包含的参数已经被正确注册，并且将被所有的Module方法都可见。
        
        '''
        self.vars = nn.ParameterList() ## 这个字典中包含了所有需要被优化的tensor
        self.vars_bn = nn.ParameterList()  
        
        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                ## [ch_out, ch_in, kernel_size, kernel_size]
                weight = nn.Parameter(torch.ones(*param[:4])) ## 产生*param大小的全为1的tensor
                torch.nn.init.kaiming_normal_(weight) ## 初始化权重
                self.vars.append(weight) ## 加到nn.ParameterList中
                
                bias = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(bias)
                
            elif name is 'linear':
                weight = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(weight)
                self.vars.append(weight)
                bias  = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(bias)
            
            elif name is 'bn':
                ## 对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作,
                ## BN层在训练过程中，会将一个Batch的中的数据转变成正态分布
                weight = nn.Parameter(torch.ones(param[0]))
                self.vars.append(weight)
                bias = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(bias)
                
                ### 
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad = False)
                running_var = nn.Parameter(torch.zeros(param[0]), requires_grad = False)
                
                self.vars_bn.extend([running_mean, running_var]) ## 在列表附加参数
                
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
                
            else:
                raise NotImplementedError       
    
    
    ## self.net(x_support[i], vars=None, bn_training = True)
    ## x: torch.Size([5, 1, 28, 28])
    ## 构造模型
    def forward(self, x, vars = None, bn_training=True):
        '''
        :param bn_training: set False to not update
        :return: 
        '''
        
        if vars is None:
            vars = self.vars
            
        idx = 0 ; bn_idx = 0
        for name, param in self.config:
            if name is 'conv2d':
                weight, bias = vars[idx], vars[idx + 1]
                x = F.conv2d(x, weight, bias, stride = param[4], padding = param[5]) 
                idx += 2
                
            elif name is 'linear':
                weight, bias = vars[idx], vars[idx + 1]
                x = F.linear(x, weight, bias)
                idx += 2
                
            elif name is 'bn':
                weight, bias = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight= weight, bias = bias, training = bn_training)
                idx += 2
                bn_idx += 2
            
            elif name is 'flatten':
                x = x.view(x.size(0), -1)
            
            elif name is 'relu':
                x = F.relu(x, inplace = [param[0]])
            
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError
            
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        
        return x
    
    
    
    def parameters(self):
        
        return self.vars