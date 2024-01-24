"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.backend import torch
import scipy.io
from scipy.interpolate import griddat
class Beltrami_flow():

    def __init__(self):
        self.a = 1
        self.d = 1
        self.Re = 1

    def pde(self,x, u):
        u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

        u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
        u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
        u_vel_z = dde.grad.jacobian(u, x, i=0, j=2)
        u_vel_t = dde.grad.jacobian(u, x, i=0, j=3)
        u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

        v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
        v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
        v_vel_z = dde.grad.jacobian(u, x, i=1, j=2)
        v_vel_t = dde.grad.jacobian(u, x, i=1, j=3)
        v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
        v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

        w_vel_x = dde.grad.jacobian(u, x, i=2, j=0)
        w_vel_y = dde.grad.jacobian(u, x, i=2, j=1)
        w_vel_z = dde.grad.jacobian(u, x, i=2, j=2)
        w_vel_t = dde.grad.jacobian(u, x, i=2, j=3)
        w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
        w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
        w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

        p_x = dde.grad.jacobian(u, x, i=3, j=0)
        p_y = dde.grad.jacobian(u, x, i=3, j=1)
        p_z = dde.grad.jacobian(u, x, i=3, j=2)

        momentum_x = (
            u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
            + p_x
            - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
        )
        momentum_y = (
            v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
            + p_y
            - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
        )
        momentum_z = (
            w_vel_t
            + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
            + p_z
            - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
        )
        continuity = u_vel_x + v_vel_y + w_vel_z

        return [momentum_x, momentum_y, momentum_z, continuity]


    def u_func(self,x):
        return (
            -a
            * (
                np.exp(a * x[:, 0:1]) * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
                + np.exp(a * x[:, 2:3]) * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def v_func(self,x):
        return (
            -a
            * (
                np.exp(a * x[:, 1:2]) * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
                + np.exp(a * x[:, 0:1]) * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def w_func(self,x):
        return (
            -a
            * (
                np.exp(a * x[:, 2:3]) * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
                + np.exp(a * x[:, 1:2]) * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def p_func(self,x):
        return (
            -0.5
            * a ** 2
            * (
                np.exp(2 * a * x[:, 0:1])
                + np.exp(2 * a * x[:, 1:2])
                + np.exp(2 * a * x[:, 2:3])
                + 2
                * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
                * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
                * np.exp(a * (x[:, 1:2] + x[:, 2:3]))
                + 2
                * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
                * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
                * np.exp(a * (x[:, 2:3] + x[:, 0:1]))
                + 2
                * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
                * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
                * np.exp(a * (x[:, 0:1] + x[:, 1:2]))
            )
            * np.exp(-2 * d ** 2 * x[:, 3:4])
        )

    def Get_Data(self):

        spatial_domain = dde.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
        temporal_domain = dde.geometry.TimeDomain(0, 10)
        spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

        boundary_condition_u = dde.icbc.DirichletBC(
            spatio_temporal_domain, self.u_func, lambda _, on_boundary: on_boundary, component=0
        )
        boundary_condition_v = dde.icbc.DirichletBC(
            spatio_temporal_domain,  self.v_func, lambda _, on_boundary: on_boundary, component=1
        )
        boundary_condition_w = dde.icbc.DirichletBC(
            spatio_temporal_domain,  self.w_func, lambda _, on_boundary: on_boundary, component=2
        )

        initial_condition_u = dde.icbc.IC(
            spatio_temporal_domain,  self.u_func, lambda _, on_initial: on_initial, component=0
        )
        initial_condition_v = dde.icbc.IC(
            spatio_temporal_domain,  self.v_func, lambda _, on_initial: on_initial, component=1
        )
        initial_condition_w = dde.icbc.IC(
            spatio_temporal_domain,  self.w_func, lambda _, on_initial: on_initial, component=2
        )

        data = dde.data.TimePDE(
            spatio_temporal_domain,
            self.pde,
            [
                boundary_condition_u,
                boundary_condition_v,
                boundary_condition_w,
                initial_condition_u,
                initial_condition_v,
                initial_condition_w,
            ],
            num_domain=50000,
            num_boundary=5000,
            num_initial=5000,
            num_test=5000,
        )

        return data


    def Save_4mat(self):

        self.data=self.Get_Data()

        x, y, z = np.meshgrid(
            np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
        )

        X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

        t_0 = np.zeros(1000).reshape(1000, 1)
        t_1 = np.ones(1000).reshape(1000, 1)

        # 提取 u, v, w, p
        physical_quantities = vars(data)["train_x_all"]
        x, y, z, t = physical_quantities[:, 0], physical_quantities[:, 1], physical_quantities[:, 2], physical_quantities[:, 3]
        u=u_func(physical_quantities)
        w=w_func(physical_quantities)
        v=v_func(physical_quantities)
        p=p_func(physical_quantities)
        # 创建一个 (10, 10, 10) 的网格坐标
        xi, yi, zi = np.meshgrid(np.linspace(min(x), max(x), 10),
                                 np.linspace(min(y), max(y), 10),
                                 np.linspace(min(z), max(z), 10))

        # 将一维数据映射到网格上
        points = np.column_stack((x, y, z))  # 一维坐标数据
        u_mapped = griddata(points, u, (xi, yi, zi), method='nearest')
        v_mapped = griddata(points, v, (xi, yi, zi), method='nearest')
        w_mapped = griddata(points, w, (xi, yi, zi), method='nearest')
        p_mapped = griddata(points, p, (xi, yi, zi), method='nearest')

        # 移除最后的维度
        u_sliced = u_mapped[:, :, :, 0]
        v_sliced = v_mapped[:, :, :, 0]
        w_sliced = w_mapped[:, :, :, 0]
        p_sliced = p_mapped[:, :, :, 0]
        print(u_sliced)
        # Prepare data for saving
        data_to_save = {
            "x": xi,
            "y": yi,
            "z": zi,
            "t": t,
            "u": u_sliced,
            "w": v_sliced,
            "v": w_sliced,
            "p": p_sliced
        }
        # Save data to .mat file
        mat_file_path = 'physical_quantities_data.mat'
        scipy.io.savemat(mat_file_path, data_to_save)
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
class PDE_BeltramiData(PDE_base):
    def __init__(self):
        self.a = 1
        self.d = 1
        self.Re = 1e5

    def torch_u(self, x, y):
        # 确保 x 和 t 是 torch.Tensor 类型
        out= torch.sin(self.k0*x)*torch.sin(self.k0*y)
        out=out.unsqueeze(1)
        return out
    def pde(self,net,data):
    
        u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

        u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
        u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
        u_vel_z = dde.grad.jacobian(u, x, i=0, j=2)
        u_vel_t = dde.grad.jacobian(u, x, i=0, j=3)
        u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

        v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
        v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
        v_vel_z = dde.grad.jacobian(u, x, i=1, j=2)
        v_vel_t = dde.grad.jacobian(u, x, i=1, j=3)
        v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
        v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

        w_vel_x = dde.grad.jacobian(u, x, i=2, j=0)
        w_vel_y = dde.grad.jacobian(u, x, i=2, j=1)
        w_vel_z = dde.grad.jacobian(u, x, i=2, j=2)
        w_vel_t = dde.grad.jacobian(u, x, i=2, j=3)
        w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
        w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
        w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

        p_x = dde.grad.jacobian(u, x, i=3, j=0)
        p_y = dde.grad.jacobian(u, x, i=3, j=1)
        p_z = dde.grad.jacobian(u, x, i=3, j=2)

        momentum_x = (
            u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
            + p_x
            - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
        )
        momentum_y = (
            v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
            + p_y
            - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
        )
        momentum_z = (
            w_vel_t
            + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
            + p_z
            - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
        )
        continuity = u_vel_x + v_vel_y + w_vel_z

        return [momentum_x, momentum_y, momentum_z, continuity]

    def beltrami_func(self,x):
        #x 有4个维度
        #u（x,y,z,t）
        a=self.a
        d=self.d
        return (
            -a
            * (
                np.exp(a * x[:, 0:1]) * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
                + np.exp(a * x[:, 2:3]) * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )
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















