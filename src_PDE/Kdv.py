
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
class PDE_KDVData(PDE_base):
    def __init__(self):
        self.k0 = 2*np.pi *1000
    
        self.data_mse=nn.MSELoss()

    def torch_u(self, x, y):
        # 确保 x 和 t 是 torch.Tensor 类型
        out= torch.sin(self.k0*x)*torch.sin(self.k0*y)
        out=out.unsqueeze(1)
        return out
    def pde(self,net,data):
        
        y_xx = dde.grad.hessian(y, x, i=0, j=0)
        y_yy = dde.grad.hessian(y, x, i=1, j=1)

        f = k0 ** 2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
        return -dy_xx - dy_yy - k0 ** 2 * y - f
    def pde_loss(self,net,pde_data):
        #data:[batch,2]
        # 确保 data 的相关列设置了 requires_grad=True  对于data：第0维度是x，t是1维度

        #train_x 里面筛选出来domian,不要bc

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
        f=self.k0 ** 2 * torch.sin(self.k0 * pde_data[:, 0:1]) * torch.sin(self.k0 * pde_data[:, 1:2])

        # 计算 PDE 残差
        pde_loss =  -ddu_ddx - ddu_ddy-self.k0**2 *u+f

        #mse
        pde_loss=torch.mean(torch.square(pde_loss))
        return pde_loss

    def bc_loss(self,net,data):
        #data:[batch,2]
        #targets=se
        # lf.torch_u(x=inputs[:,0],t=inputs[:,1])  # 创建一个与u形状相同且元素为1的张量(,1)
        inputs=data
        outputs=net(data) # 计算网络输出
        #标记bc序列
        bcs_start = np.cumsum([0] + self.data.num_bcs)
        bcs_start = list(map(int, bcs_start))

        losses= []
        for i, bc in enumerate(self.data.bcs): #ic and bc
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.train_x有序
            error = bc.error(inputs,inputs,outputs, beg, end)

            #求mse
            error_scalar = torch.mean(torch.square(error))
            
            losses.append(error_scalar)
        # 将losses列表转换为Tensor
        losses_tensor = torch.stack(losses)
        bc_mse = torch.mean(losses_tensor)

        return bc_mse

    def data_loss(self,net,data):
        #data:[batch,2]
        u = net(data)  # 计算网络输出
        # 计算 MSE 损失
        targets=self.torch_u(x=data[:,0],y=data[:,1])  # 创建一个与u形状相同且元素为1的张量(,1)
        data_loss = self.data_mse(u,targets)
        return data_loss
    def boundary(self,_,on_boundary):

        return on_boundary

    def Get_Data(self)->dde.data.PDE: #u(x,t)
        self.geom = dde.geometry.Rectangle([0, 0], [0.5, 0.5])
        

    
        bc = dde.icbc.DirichletBC(self.geom, lambda x: 0, self.boundary)
        self.data = dde.data.PDE(
            self.geom,
            self.pde,
            bc,
            num_domain=10000,#sqrt(3600)=60
            num_boundary=3600,
            num_test=3600,
            train_distribution="pseudo",
        )

        return self.data

    def helomotz_eq_exact_solution(self,x, y):
        """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

        Parameters
        ----------
        x : np.ndarray
        t : np.ndarray
        """
        print("x.shape",x.shape)
        
        usol=np.sin(self.k0*x)*np.sin(self.k0*y)

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

        ax.scatter(x,y, c=u,cmap=cmap,s=1)

        ax.set_xlabel("x")
        
        ax.set_ylabel("y")

        ax.set_title(title)
        return u
    def plot_pred(self,ax=None,model=None,
                    title="Pred u(x,y)",
                    cmap="bwr",data=None):
        # Number of points in each dimension:
        # 提取 x 和 t
        x = data[:, 0]
        y = data[:, 1]
        data=torch.from_numpy(data).float()

        # 获取 usol 值
        usol_net = model(data).detach().numpy()

        # 绘制热力图
        ax.scatter(x,y, c=usol_net, cmap=cmap,s=1)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        return usol_net

    def train(self,net=None):
        pass


if __name__ == "__main__":
    Helmholtz=PDE_HelmholtzData()
    data=Helmholtz.Get_Data()
    if (type(data)==dde.data.PDE):
        print(vars(data))
    data.train_points()
    import numpy as np
    import matplotlib.pyplot as plt









