
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
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def Get_Data(self):
        pass
    def torch_u(self,x,t):
        pass
    def pde(self,net,data):
        pass
    def train(self,net=None):
        pass
class PDE_HeatData(PDE_base):
    def __init__(self,freq=1.0):
        self.a = 0.4
        self.L = 1
        self.n = freq
        self.data_mse=nn.MSELoss()

    def torch_u(self, x, t):
        # 确保 x 和 t 是 torch.Tensor 类型
        out= torch.exp(-(self.n ** 2 * np.pi ** 2 * self.a * t) / (self.L ** 2)) * torch.sin(
            self.n * np.pi * x / self.L)
        out=out.unsqueeze(1)
        return out
    def pde(self,net,data):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - self.a * dy_xx
    def pde_loss(self,net,pde_data):
        #data:[batch,2]
        # 确保 data 的相关列设置了 requires_grad=True  对于data：第0维度是x，t是1维度

        #train_x 里面筛选出来domian,不要bc

        u=net(pde_data)  # 计算网络输出
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
        lfh =  du_dt 
        #mse
        loss=pde_loss(lfh,self.a * ddu_ddx)
        return loss

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
        targets=self.torch_u(x=data[:,0],t=data[:,1])  # 创建一个与u形状相同且元素为1的张量(,1)
        data_loss = self.data_mse(u,targets)
        return data_loss

    def Get_Data(self,**kwagrs)->dde.data.TimePDE: #u(x,t)
        
        
        self.geom = dde.geometry.Rectangle([0, 0], [1, 1])
        domain_numbers=kwagrs["domain_numbers"]
        boundary_numbers=kwagrs["boundary_numbers"]
        test_numbers=kwagrs["test_numbers"]
        
        self.geom = dde.geometry.Interval(0, self.L)
        timedomain = dde.geometry.TimeDomain(0, 1)
        
        geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
        ic = dde.icbc.IC(
            geomtime,
            lambda x: torch.sin(self.n * np.pi * x[:, 0:1] / self.L),
            lambda _, on_initial: on_initial,
        )
        self.data = dde.data.TimePDE(
            geomtime,
            self.pde,
            [bc, ic],
            num_domain=domain_numbers,#sqrt(3600)=60
            num_boundary=boundary_numbers,
            num_initial=1000,
            num_test=test_numbers,
            train_distribution="pseudo",
        )

        return self.data

    def heat_eq_exact_solution(self,x, t):
        """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

        Parameters
        ----------
        x : np.ndarray
        t : np.ndarray
        """
        return np.exp(-(self.n ** 2 * np.pi ** 2 * self.a * t) / (self.L ** 2)) * np.sin(self.n * np.pi * x / self.L)

    def gen_exact_solution(self,x,t):
        """Generates exact solution for the heat equation for the given values of x and t."""

        usol= self.heat_eq_exact_solution(x, t)

        return usol

    def plot_exact( self,
                    ax=None,
                    title="Exact u(x,y)", 
                    cmap="bwr",data=None):
        x = data[:, 0]
        t = data[:, 1]

        u=self.gen_exact_solution(x,t)
        print("exact",u.shape)

        sc=ax.scatter(t,x, c=u,cmap="bwr",s=1)

        ax.set_xlabel("t")
        
        ax.set_ylabel("x")
           
       
        cb=plt.colorbar(sc, ax=ax)

        ax.set_title("true u(x,t)")
        return u,cb
    def plot_pred(self,ax=None,model=None,
                    title="Pred u(x,y)",
                    cmap="bwr",data=None):
        # Number of points in each dimension:
        # 提取 x 和 t
        x = data[:, 0]
        t = data[:, 1]
        data=torch.from_numpy(data).float()

        # 获取 usol 值
        if self.device =="cuda":
            usol_net = model(data).cpu().detach().numpy()
            
        else:
            usol_net = model(data).cpu().detach().numpy()

        # 绘制热力图
        sc=ax.scatter(t,x, c=usol_net, cmap="bwr",s=1)
        cb= plt.colorbar(sc, ax=ax)

        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title("pred u(x,t)")
        return usol_net,cb
    def train(self,net=None):
        pass


if __name__ == "__main__":
    heat=PDE_HeatData()
    data=heat.Get_Data()
    if (type(data)==dde.data.TimePDE):
        print(vars(data))
    data.train_points()











