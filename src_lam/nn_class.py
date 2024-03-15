
import torch
import torch.nn as nn
from  .act_fuction import *
import numpy as np
import copy
from .MOE import *
#mlp
class Single_MLP(nn.Module):
    def __init__(self,
                 input_size,
                 layer_set:list,#[1,150,150,150,1] means 4 linear layers and 3 act
                 use_residual:list,
                 **kwargs):
        super(Single_MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.use_residual = use_residual
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

# Define the gating model
class Gating(nn.Module):
   def __init__(self, input_dim,
               num_experts, dropout_rate=0.1):
      super(Gating, self).__init__()

      # Layers
      self.layer1 = nn.Linear(input_dim, 128)
      self.dropout1 = nn.Dropout(dropout_rate)

      self.layer2 = nn.Linear(128, 256)
      self.leaky_relu1 = nn.LeakyReLU()
      self.dropout2 = nn.Dropout(dropout_rate)

      self.layer4 = nn.Linear(128, num_experts)

   def forward(self, x):
      x = torch.relu(self.layer1(x))
      x = self.dropout1(x)

      x = self.layer2(x)
      x = self.leaky_relu1(x)
      x = self.dropout2(x)

      x = self.layer3(x)
      x = self.leaky_relu2(x)
      x = self.dropout3(x)

      return torch.softmax(self.layer4(x), dim=1)

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
        scale_learn = kwargs["scale_learn"]#是否学习尺度系数
        self.scale_number=len(scale_number)
        
        if "Siren" in act_set: #siren
            
        
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
            
         # 计算并打印参数总量
        total_params = sum(p.numel() for p in self.Multi_scale.parameters())
        print(f'Total number of parameters: {total_params}')
            
        

    def forward(self, x):
        """定义前向传播。"""
        y=[]
        
        # Calculate the expert outputs
        outputs = torch.stack(
            [subnet(self.sub_omegas[index]*x) for index,subnet in enumerate(self.Multi_scale)], dim=2)
        

        out= torch.sum(outputs, dim=2)
        
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

class MoE_Multi_Scale(nn.Module):

    def __init__(self,
                 sub_layer_number,# 子网络个数
                 layer_set,#[1,150,150,150,1] means 4 linear layers and 3 act
                 act_set,#激活函数 4*3 4个子网络，每个子网络3层
                 ini_set,#  4个子网络，每个1个初始化方法
                 scale_number:list,# 4个子网络，每个1个尺度系数
                 residual = False,
                 **kwargs):
        super().__init__()
        scale_learn = kwargs["scale_learn"]#是否学习尺度系数
        self.scale_number=len(scale_number)
        #专家数量
        num_experts = self.scale_number
        #k专家
        k= kwargs["sparse_experts"]
        print("sparse_k",k)

        act_list= self._Return_act_list(act_set) #4个激活函数,module list,每个有3个激活函数
        kernel_method=self._Return_init_list(ini_set)#4个初始化方法，每个1个初始化方法
        
        print("scale_learn",scale_learn)

        one_layer = Single_MLP(     input_size=layer_set[0],
                                    layer_set=layer_set,
                                    use_residual=residual,
                                    activation_set=act_list[0])
            
        self.sub_omegas=scale_number
        
        if scale_learn:# 可学习
            self.sub_omegas = torch.nn.Parameter(torch.tensor(self.sub_omegas, dtype=torch.float))
        else:
            self.sub_omegas = torch.tensor(self.sub_omegas, dtype=torch.float)
            

        self.Moe_scale = MoE (subnet=one_layer,input_size = layer_set[0],output_size=1
                              ,num_experts = num_experts,scale_coeff = self.sub_omegas,
                              k= k)
        # 计算并打印参数总量
        total_params = sum(p.numel() for p in self.Moe_scale.parameters())
        print(f'Total number of parameters: {total_params}')



    def forward(self, x):

        """定义前向传播。"""
        out,loss,gates = self.Moe_scale(x)
        
        return out,loss,gates


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
    print("hi1")

    





