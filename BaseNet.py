import  torch
from    torch import nn
from    torch.nn import functional as F

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.vars = nn.ParameterList()  ## 包含了所有需要被优化的tensor
        self.vars_bn = nn.ParameterList()
        
        # 第1个conv2d
        weight = nn.Parameter(torch.ones(64, 1, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        # 第1个BatchNorm层
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        running_mean = nn.Parameter(torch.zeros(64), requires_grad= False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad= False)
        self.vars_bn.extend([running_mean, running_var])
        
        # 第2个conv2d
        weight = nn.Parameter(torch.ones(64, 64, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        # 第2个BatchNorm层
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        running_mean = nn.Parameter(torch.zeros(64), requires_grad= False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad= False)
        self.vars_bn.extend([running_mean, running_var])
        
        # 第3个conv2d
        weight = nn.Parameter(torch.ones(64, 64, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        # 第3个BatchNorm层
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        running_mean = nn.Parameter(torch.zeros(64), requires_grad= False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad= False)
        self.vars_bn.extend([running_mean, running_var])
        
        # 第4个conv2d
        weight = nn.Parameter(torch.ones(64, 64, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        # 第4个BatchNorm层
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        running_mean = nn.Parameter(torch.zeros(64), requires_grad= False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad= False)
        self.vars_bn.extend([running_mean, running_var])
        
        ##linear
        weight = nn.Parameter(torch.ones([5,64]))
        bias = nn.Parameter(torch.zeros(5))
        self.vars.extend([weight,bias])
        
    def forward(self, x, params = None, bn_training=True):
        '''
        :bn_training: set False to not update
        :return: 
        '''
        if params is None:
            params = self.vars
        
        weight, bias = params[0], params[1]  # 第1个CONV层
        x = F.conv2d(x, weight, bias, stride = 1, padding = 1)
        weight, bias = params[2], params[3]  # 第1个BN层
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
        x = F.batch_norm(x, running_mean, running_var, weight=weight,bias =bias, training= bn_training)
        x = F.relu(x, inplace = [True])  #第1个relu
        x = F.max_pool2d(x,kernel_size=2)  #第1个MAX_POOL层  
        
        
        
        weight, bias = params[4], params[5]  # 第2个CONV层
        x = F.conv2d(x, weight, bias, stride = 1, padding = 1)
        weight, bias = params[6], params[7]  # 第2个BN层
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=weight,bias =bias, training= bn_training)
        x = F.relu(x, inplace = [True])  #第2个relu
        x = F.max_pool2d(x,kernel_size=2)  #第2个MAX_POOL层   
        
        
        weight, bias = params[8], params[9]  # 第3个CONV层
        x = F.conv2d(x, weight, bias, stride = 1, padding = 1)
        weight, bias = params[10], params[11]  # 第3个BN层
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        x = F.batch_norm(x, running_mean, running_var, weight=weight,bias =bias, training= bn_training)
        x = F.relu(x, inplace = [True])  #第3个relu
        x = F.max_pool2d(x,kernel_size=2)  #第3个MAX_POOL层
        
        
        weight, bias = params[12], params[13]  # 第4个CONV层
        x = F.conv2d(x, weight, bias, stride = 1, padding = 1)
        weight, bias = params[14], params[15]  # 第4个BN层
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        x = F.batch_norm(x, running_mean, running_var, weight=weight,bias =bias, training= bn_training)
        x = F.relu(x, inplace = [True])  #第4个relu
        x = F.max_pool2d(x,kernel_size=2)  #第4个MAX_POOL层
        
        
        
        x = x.view(x.size(0), -1) ## flatten
        weight, bias = params[-2], params[-1]  # linear
        x = F.linear(x, weight, bias)
        
        output = x
        
        return output
    
    
    def parameters(self):
        
        return self.vars
