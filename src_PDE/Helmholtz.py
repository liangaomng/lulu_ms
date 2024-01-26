
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
import matplotlib as mpl
class  PDE_base():
    def __init__(self):
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def Get_Data(self):
        pass
    def torch_u(self,x,t):
        pass
    def pde(self,net,data):
        pass
    def train(self,net=None):
        pass
class PDE_HelmholtzData(PDE_base):
    def __init__(self,k=1):
        self.k0 = k*np.pi 
        self.data_mse=nn.MSELoss()


    # Redefine the functions to accept a point (x, y) instead of a vector x
    def np_f(self,x, y):
        return self.k0**2*torch.sin(self.k0*x)*torch.sin(self.k0*y)
        
    def torch_f(self,x, y):
        
        return  self.k0**2*torch.sin(self.k0*x)*torch.sin(self.k0*y)
    
    def np_u(self,x, y):
        
        return np.sin(self.k0 * x) *np.sin(self.k0 * y)
    
    def torch_u(self, x, y):
        
        return torch.sin(self.k0*x)*torch.sin(self.k0*y)
    def pde(self,x,y):
        
        
        y_xx = dde.grad.hessian(y, x, i=0, j=0)
        y_yy = dde.grad.hessian(y, x, i=1, j=1)
        
        return 0
       

        
    def pde_loss(self,net,pde_data):
        #data:[batch,2]
        # 确保 data 的相关列设置了 requires_grad=True  对于data：第0维度是x，t是1维度

        u=net(pde_data)  # 计算网络输出
        grad_outputs = torch.ones_like(u)  # 创建一个与u形状相同且元素为1的张量

        # 计算一阶导数
        du_data=grad(u,pde_data,grad_outputs,create_graph=True)[0]
        du_dy=du_data[:,1].unsqueeze(1)
        du_dx=du_data[:,0].unsqueeze(1)
        # 计算二阶导数
        ddu_ddata=grad(du_dx,pde_data,grad_outputs, create_graph=True)[0]
        ddu_ddx=ddu_ddata[:,0]
        ddu_ddy=grad(du_dy,pde_data,grad_outputs, create_graph=True)[0]
        ddu_ddy=ddu_ddy[:,1]
        
      
        ddu_ddx=ddu_ddx.reshape(-1,1)
        ddu_ddy=ddu_ddy.reshape(-1,1)
        lfh=ddu_ddx+ddu_ddy+(self.k0**2)*u
    
        f=self.torch_f(x=pde_data[:,0:1],y=pde_data[:,1:2])

    
        loss =torch.mean((lfh+f)*(lfh+f)*0.5)

        return loss

    def bc_loss(self,net,data):
        #data:[batch,2]
        #targets=se
        # lf.torch_u(x=inputs[:,0],t=inputs[:,1])  # 创建一个与u形状相同且元素为1的张量(,1)
        #case 边界0
        outputs=net(data) # 计算网络输出
        bc_mse = self.data_mse(outputs, torch.zeros_like(outputs))
        return bc_mse


    def data_loss(self,net,data):
        #data:[batch,2]
        u = net(data)  # 计算网络输出
        # 计算 MSE 损失
        targets=self.torch_u(x=data[:,0:1],y=data[:,1:2])  # 创建一个与u形状相同且元素为1的张量(,1)
        data_loss = self.data_mse(u,targets)
        return data_loss
    def boundary(self,_,on_boundary):

        return on_boundary

    def Get_Data(self,**kwagrs)->dde.data.PDE: #u(x,t)
        
        self.geom = dde.geometry.Rectangle([0, 0], [1, 1])
        domain_numbers=kwagrs.get("domain_numbers",300)
        boundary_numbers=kwagrs.get("boundary_numbers",10)
        test_numbers=kwagrs.get("test_numbers",1000)
        
        
        bc = dde.icbc.DirichletBC(self.geom, lambda x: 0, self.boundary)
        self.data = dde.data.PDE(
            self.geom,
            self.pde,
            bc,
            num_domain=domain_numbers,#sqrt(3600)=60
            num_boundary=boundary_numbers,
            num_test=test_numbers,
            train_distribution="uniform",
        
        )

        return self.data

    def helomotz_eq_exact_solution(self,x, y):
        """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

        Parameters
        ----------
        x : np.ndarray
        t : np.ndarray
        """
    
        usol=self.np_u(x,y)

        # 检查 x 和 y 是否靠近边界 (例如，靠近 0 或 1)
        for i in range (len(x)):
             if self.geom.on_boundary((x[i], y[i])):
                usol[i] = 0
        return usol

    def gen_exact_solution(self,x,y):
        """Generates exact solution for the heat equation for the given values of x and t."""

        usol= self.helomotz_eq_exact_solution(x, y)
    
        return usol

    def plot_exact( self,
                    ax=None,
                    title="Exact u(x,y)", 
                    cmap="bwr",data=None):
        x = data[:, 0]
        y = data[:, 1]

        u=self.gen_exact_solution(x,y)

        sc=ax.scatter(x,y, c=u,cmap="bwr",s=1)

        ax.set_xlabel("x")
        
        ax.set_ylabel("y")
           
        # 检查并删除已存在的颜色条
        cb=plt.colorbar(sc, ax=ax)

        ax.set_title(title)
        return u,cb
    def plot_pred(self,ax=None,model=None,
                    title="Pred u(x,y)",
                    cmap="bwr",data=None):
        # Number of points in each dimension:
        # 提取 x 和 t
        x = data[:, 0]
        y = data[:, 1]
        data=torch.from_numpy(data).float()

         # 获取 usol 值
        if self.device =="cuda":
             usol_net = model(data).cpu().detach().numpy()
            
        else:
            usol_net = model(data).cpu().detach().numpy()
            
        # 绘制热力图
        sc=ax.scatter(x,y, c=usol_net, cmap="bwr",s=1)
        cb= plt.colorbar(sc, ax=ax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        return usol_net,cb


if __name__ == "__main__":
    
    
    # 生成数据
    hel=PDE_HelmholtzData()
    data=hel.Get_Data()
    if (type(data)==dde.data.PDE):
        pass
    coord=data.train_x
    print(data.train_x_all)

    
    