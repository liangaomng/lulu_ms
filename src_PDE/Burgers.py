
import deepxde.geometry as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.FloatTensor)
#https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/heat.html
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import grad
class  PDE_base():
    def __init__(self):
        pass
       
    def Get_Data(self):
        pass
    def torch_u(self,x,t):
        pass
    def pde(self,net,data):
        pass
    def train(self,net=None):
        pass

def gen_testdata():
    data = np.load("src_PDE/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    return t,x,exact

class PDE_BurgersData(PDE_base):
    def __init__(self,v=0.01/np.pi):
        self.v = v
        self.data_mse=nn.MSELoss()
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name ="Burgers"
    def torch_u(self):
        # 确保 x 和 t 是 torch.Tensor 类型
        t,x,u = gen_testdata()
        torch_out = torch.from_numpy(u).float()   
         #out  是（100，256）t是100，x是256
        return t,x,torch_out
    def pde(x, y):
      dy_x = dde.grad.jacobian(y, x, i=0, j=0)
      dy_t = dde.grad.jacobian(y, x, i=0, j=1)
      dy_xx = dde.grad.hessian(y, x, i=0, j=0)
      return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    def pde_loss(self,net,pde_data,MOE=False):
        #data:[batch,2]
        # 确保 data 的相关列设置了 requires_grad=True  对于data：第0维度是x，t是1维度

        #train_x 里面筛选出来domian,不要bc
         
        u = net(pde_data)  # 计算网络输出
        
        grad_outputs = torch.ones_like(u)  # 创建一个与u形状相同且元素为1的张量

        # 计算一阶导数
        du_data=grad(u,pde_data,grad_outputs,create_graph=True)[0]
        du_dt=du_data[:,1]
        du_dx=du_data[:,0].unsqueeze(1)
        # 计算二阶导数
        ddu_ddata=grad(du_dx,pde_data,grad_outputs, create_graph=True)[0]
        ddu_ddx=ddu_ddata[:,0]
        pde_loss =  nn.MSELoss()    
        # 计算 PDE 残差
        lfh =  du_dt  + du_dx.squeeze()  * u.squeeze() 
        #mse
        loss=pde_loss(lfh,self.v * ddu_ddx)
        
        #
        if MOE==True:
            load,w_gates,gates,moe_loss = net.Moe_model._record_load()

        else:
            moe_loss = torch.tensor(0).to(self.device)
            gates = torch.tensor(0).to(self.device)
            w_gates = torch.tensor(0).to(self.device)
            load = torch.tensor(0).to(self.device)
            
        return loss,moe_loss,gates,w_gates,load

    def bc_loss(self,net,data,MOE=False):
        #data:[batch,2]
        #targets=se
        # lf.torch_u(x=inputs[:,0],t=inputs[:,1])  # 创建一个与u形状相同且元素为1的张量(,1)
        inputs= data
        output=net(data) # 计算网络输出

        #标记bc序列
        bcs_start = np.cumsum([0] + self.data.num_bcs)
        bcs_start = list(map(int, bcs_start))

        losses= []
        for i, bc in enumerate(self.data.bcs): #ic and bc
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.train_x有序
            error = bc.error(inputs,inputs,output, beg, end)
            #求mse
            error_scalar = torch.mean(torch.square(error))
            
            losses.append(error_scalar)
        # 将losses列表转换为Tensor
        losses_tensor = torch.stack(losses)
        bc_mse = torch.mean(losses_tensor)
        
        if MOE ==True:
            load,w_gates,gates,moe_loss = net.Moe_model._record_load()
        else:
            moe_loss = torch.tensor(0).to(self.device)
            load = torch.tensor(0).to(self.device)
            gates = torch.tensor(0).to(self.device)
            w_gates = torch.tensor(0).to(self.device)
            
        return bc_mse,moe_loss,gates,w_gates,load

    def data_loss(self,net,data,MOE=False):
        #data:[batch,2]
        # 计算 MSE 损失
        t,x,targets = self.torch_u()  

        targets = targets.to(self.device)
        T, X = np.meshgrid(t, x,indexing="ij")

        # 将网格展平，以便每一行都是一个(t, x)对
        TX_combinations = np.vstack([ X.ravel(),T.ravel(),]).T
        TX_combinations = torch.from_numpy(TX_combinations).to(self.device).float()

        nn_u = net(TX_combinations)


        targets = targets.reshape(-1,1)

        data_loss = torch.mean((nn_u-targets)**2)
    
        if MOE ==True:
            load,w_gates,gates,moe_loss = net.Moe_model._record_load()

        else:
            moe_loss = torch.tensor(0).to(self.device)
            load = torch.tensor(0).to(self.device)
            gates = torch.tensor(0).to(self.device)
            w_gates = torch.tensor(0).to(self.device)
            
            


        return data_loss,moe_loss,gates,w_gates,load

    def Get_Data(self,**kwagrs)->dde.data.TimePDE: #u(x,t)
        
        
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 0.99)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
        domain_numbers=kwagrs.get("domain_numbers",6400)
        boundary_numbers=kwagrs.get("boundary_numbers",6400)
        test_numbers=kwagrs.get("test_numbers",1600)
    
    
        bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
        ic = dde.icbc.IC(
            geomtime, lambda x: -torch.sin(torch.pi * x[:, 0:1]), lambda _, on_initial: on_initial
        )
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 0.99)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        self.data = dde.data.TimePDE(
                geomtime, 
                self.pde, [bc, ic], 
                num_domain=1600, 
                num_boundary=1600,
                num_initial=1000,
                num_test=test_numbers,
                train_distribution="pseudo",
        )

        return self.data


    def plot_exact( self,
                    ax=None,
                    cmap="bwr",
                    title="Exact u(x,y)", 
                    data=None,
                   ):

        t,x,u = self.torch_u()

        u = u.numpy()
        # Make sure t and x are 1D arrays for meshgrid

        # Create a 2D grid of t and x
        T, X = np.meshgrid(t, x, indexing='ij')

        # Flatten the exact values to match the meshgrid
        exact_flat = u.flatten()
        T=T.flatten()
        X=X.flatten()
        # Scatter plot
        sc=ax.scatter(T.flatten(), X.flatten(), c=exact_flat,s=10,cmap="bwr",vmin=-1,vmax=1)
       
        ax.set_xlabel('t',fontsize=14)
        ax.set_ylabel('x',fontsize=14)
        
        # 添加colorbar
        cbar = plt.colorbar(sc, ax=ax)
        ax.set_title(title,fontsize=18)
        u =u.flatten() 
        return u,cbar,T,X
    def plot_pred(self,ax=None,model=None,
                    title="Pred u(x,y)",
                    cmap="bwr",data=None,**kwagrs):
        # Number of points in each dimension:
        # 提取 x 和 ts
        t,x,_=self.torch_u()  
        MOE = kwagrs["MOE"]
        T, X = np.meshgrid(t, x,indexing='ij')

        # 将网格展平，以便每一行都是一个(t, x)对
        TX_combinations = np.vstack([ X.ravel(),T.ravel(),]).T
        TX_combinations = torch.from_numpy(TX_combinations).float()

        TX_combinations=TX_combinations.to(self.device)
    
        
    
        # 获取 usol 值
        if self.device =="cuda":

            if MOE == False:
                usol_net=model(TX_combinations).cpu().detach().numpy()
            else: #moe     需要第一个
                usol_net=model(TX_combinations).cpu().detach().numpy()
    
          
        else:
            if MOE == False:
                usol_net=model(TX_combinations).cpu().detach().numpy()
            else: #moe     需要第一个
                usol_net=model(TX_combinations).cpu().detach().numpy()

        # 绘制热力图
        sc=ax.scatter(T.flatten(),X.flatten(),c=usol_net,cmap="bwr",s=10,vmin=-1,vmax=1)

        
        cb= plt.colorbar(sc, ax=ax)

        ax.set_xlabel("t",fontsize=14)
        ax.set_ylabel("x",fontsize=14)
        ax.set_title("pred u(x,t)",fontsize=18)
        usol_net=usol_net.flatten()
        return usol_net,cb
    def train(self,net=None):
        pass


if __name__ == "__main__":  
    burgers=PDE_BurgersData()
    data=burgers.Get_Data()












