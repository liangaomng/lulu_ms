
import torch
import torch.nn as nn
from  .act_fuction import *
import numpy as np
import copy
#mlp
class Single_MLP(nn.Module):
    def __init__(self,
                 input_size,
                 layer_set:list,#[1,150,150,150,1] means 4 linear layers and 3 act
                 use_residual=list,
                 **kwargs):
        super(Single_MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.use_residual = use_residual[0]
        layer_depth = len(layer_set)-1
        layer_width=layer_set[1]
        assert len(kwargs["activation_set"])==layer_depth -1 #激活比linear 少一个
        # 构建网络层
        for i in range(layer_depth):
            #输入
            if i == 0:
                self.layers.append(
                    nn.Linear(input_size, layer_width)
                )
                self.layers.append(
                    kwargs["activation_set"][i]
                )
            #输出
            elif i==layer_depth-1:
                self.layers.append(
                              nn.Linear(layer_width, 1)
                )

            # 中间
            else:
                self.layers.append(
                    nn.Linear(layer_width, layer_width)

                )
                self.layers.append(
                    kwargs["activation_set"][i]
                )

    def forward(self, x):
        identity = x
        for i, layer in enumerate(self.layers):
            # 对于除了第一层之外的每个激活层后，应用残差连接
            if not isinstance(layer, nn.Linear) and i > 1 and self.use_residual == True:
                # 由于输入和输出维度相同，可以直接添加
                x = layer(x) + identity

                identity = x  # 更新 identity
            else:
                x = layer(x)

                if i % 2 == 0:  # 在每个线性层后更新 identity
                    identity = x

        return x


class Multi_scale2(nn.Module):
    """Fully-connected neural network."""
    def __init__(self,
                 sub_layer_number,# 子网络个数
                 layer_set,#[1,150,150,150,1] means 4 linear layers and 3 act
                 act_set,#激活函数 4*3 4个子网络，每个子网络3层
                 ini_set,#  4个子网络，每个1个初始化方法
                 scale_number:list,# 4个子网络，每个1个尺度系数
                 residual = False,
                 **kwargs):
        super().__init__()
        scale_learn=kwargs["scale_learn"]#是否学习尺度系数
        self.scale_number=len(scale_number)
        
        if "Siren" in act_set: #siren
            print(act_set)
        
            one_layer = Siren( in_features=layer_set[0],
                               out_features=layer_set[-1],
                               hidden_features=layer_set[1],
                               hidden_layers=len(layer_set)-1,
                               outermost_linear=True
                              )
            self.Multi_scale=self._clones(one_layer,sub_layer_number) #复制4个网络
        else:
            act_list= self._Return_act_list(act_set) #4个激活函数,module list,每个有3个激活函数
            kernel_method=self._Return_init_list(ini_set)#4个初始化方法，每个1个初始化方法
            
            print("scale_learn",scale_learn)

            one_layer = Single_MLP(     input_size=layer_set[0],
                                        layer_set=layer_set,
                                        use_residual=residual,
                                        activation_set=act_list[0])
            self.Multi_scale=self._clones(one_layer,sub_layer_number) #复制4个网络
            self._init4weights(kernel_method) #为了给每个网络，初始化权重

      

        self.sub_omegas=scale_number
        
        if scale_learn:# 可学习
            self.sub_omegas = torch.nn.Parameter(torch.tensor(self.sub_omegas, dtype=torch.float))
        else:
            self.sub_omegas = torch.tensor(self.sub_omegas, dtype=torch.float)
  
            
            
    

    def forward(self, x):
        """定义前向传播。"""
        y=[]

       
        for index, subnet in enumerate(self.Multi_scale):

            y.append(subnet(self.sub_omegas[index]*x))#

      
        out=torch.sum(torch.stack(y, dim=0), dim=0) 

    
    
        return out

    def _Return_act_list(self,activation_set):
        #类方法，返回激活函数列表
        return St_act_in_4_subnet_space.activations_get(activation_set)
    def _Return_init_list(self,ini_set):

        return St_act_in_4_subnet_space.weight_ini_method(ini_set)

    def  _clones(self,module, N):
        #重复N次，深拷贝
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    def _init4weights(self,kernal_method):
        #初始化权重
        for i,m in enumerate(self.Multi_scale):
            if isinstance(m, nn.Linear):
                kernal_method[i](m.weight)
                nn.init.zeros_(m.bias)
if __name__=="__main__":
    pass
    #test





