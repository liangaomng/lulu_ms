
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
import torch

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
class PDE_PossionData(PDE_base):
    def __init__(self,mu=15,shape="square"):
        self.mu = mu
        self.d=2
        self.data_mse=nn.MSELoss()
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shape =shape

    # Redefine the functions to accept a point (x, y) instead of a vector x
    def np_f(self,x, y):
        return 4 * self.mu**2 * x**2 * np.sin(self.mu * x**2) - 2 * self.mu * np.cos(self.mu * x**2) + \
            4 * self.mu**2 * y**2 * np.sin(self.mu * y**2) - 2 * self.mu * np.cos(self.mu * y**2)
            
    def torch_f(self,x, y):
        
        return  4 * self.mu**2 * x**2 * torch.sin(self.mu * x**2) - 2 * self.mu * torch.cos(self.mu * x**2) + \
            4 * self.mu**2 * y**2 * torch.sin(self.mu * y**2) - 2 * self.mu * torch.cos(self.mu * y**2)
    def np_u(self,x, y):
        
        return np.sin(self.mu * x**2) + np.sin(self.mu * y**2)
    def torch_u(self, x, y):
        
        return torch.sin(self.mu * x**2) + torch.sin(self.mu * y**2)
 
        
    def pde(self,x,y):
        pass

        """_ritz_loss
      
    # def pde_loss(self,net,pde_data):
    
        #ritz_loss

    #     output1=net(pde_data)
    #     dfdx=grad(output1,pde_data,grad_outputs=torch.ones_like(output1),create_graph=True)[0]
    #     dfdy=grad(output1,pde_data,grad_outputs=torch.ones_like(output1),create_graph=True)[0]
        

    #     fTerm=self.torch_f(x=pde_data[:,0:1],y=pde_data[:,1:2])

        
    #     new_loss=0.5*torch.mean((dfdx*dfdx)+(dfdy*dfdy))-torch.mean(fTerm*output1)
        
    #     return new_loss
          """
      
        

    def pde_loss(self,net,pde_data):
        #data:[batch,2]
        # 确保 data 的相关列设置了 requires_grad=True  对于data：第0维度是x，t是1维度


        outputs = net(pde_data)  # 计算网络输出
        
        # 检查输出结果
        if isinstance(outputs, tuple):
            #moe 有3个输出
            u, moe_loss, gates = outputs

        else:
            # 如果输出不是元组，假设它是单一输出
            num_outputs = 1
            u=outputs
            moe_loss=0
            gates=0

        grad_outputs = torch.ones_like(u)  # 创建一个与u形状相同且元素为1的张量
 
        # 计算一阶导数
        du_data= grad(u,pde_data,grad_outputs,create_graph=True)[0]
        du_dy=du_data[:,1].unsqueeze(1)
        du_dx=du_data[:,0].unsqueeze(1)
        # 计算二阶导数
        ddu_ddata=grad(du_dx,pde_data,grad_outputs, create_graph=True)[0]
        ddu_ddx=ddu_ddata[:,0]
        ddu_ddy=grad(du_dy,pde_data,grad_outputs, create_graph=True)[0]
        ddu_ddy=ddu_ddy[:,1]
      
        
        f=self.torch_f(x=pde_data[:,0:1],y=pde_data[:,1:2])
        
        ddu_ddx=ddu_ddx.reshape(-1,1)
        ddu_ddy=ddu_ddy.reshape(-1,1)
        lfh=ddu_ddx+ddu_ddy 

        
        # 计算 PDE 残差
 
        loss = torch.mean((lfh+f)*(lfh+f))+moe_loss

        return loss,moe_loss,gates

    def bc_loss(self,net,data):
        #data:[batch,2]
        #targets=se
        # lf.torch_u(x=inputs[:,0],t=inputs[:,1])  # 创建一个与u形状相同且元素为1的张量(,1)
        inputs=data

        outputs=net(data) # 计算网络输出
         # 检查输出结果
        if isinstance(outputs, tuple):
            #moe 有3个输出
            output, moe_loss, gates = outputs
           
        else:
            # 如果输出不是元组，假设它是单一输出
         
            num_outputs = 1
            output=outputs
            moe_loss=0
            gates=0
        
        
        # 计算 MSE 损失
        bc_mse = self.data_mse(output, self.torch_u(x=inputs[:,0:1],y=inputs[:,1:2])) + moe_loss
        

        return bc_mse,moe_loss,gates

    def data_loss(self,net,data):
        #data:[batch,2]
        outputs = net(data)  # 计算网络输出
         # 检查输出结果
        if isinstance(outputs, tuple):
            #moe 有3个输出
            u, moe_loss, gates = outputs

        else:
            # 如果输出不是元组，假设它是单一输出
            num_outputs = 1
            u = outputs
            moe_loss=0
            gates=0
        
        # 计算 MSE 损失
        targets=self.torch_u(x=data[:,0:1],y=data[:,1:2]) + moe_loss  # 创建一个与u形状相同且元素为1的张量(,1)
        

        data_loss = self.data_mse(u,targets)
        return data_loss,moe_loss,gates
    
    def boundary(self,_,on_boundary):

        return on_boundary
    def five_point_star_shape(self,theta):
        """
        Defines the radius function for a regular five-pointed star
        as a function of angle theta.
        """
        # Convert polar to cartesian coordinates to use mod
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Every other vertex is further from the center
        # Adjust these values to change the star's appearance
        radius_inner = 0.5  # Radius for inner vertices
        radius_outer = 1.0  # Radius for outer vertices
        
        # Determine whether we are closer to an inner or an outer vertex
        # Adjust the modulus operation to match five segments (360/5 = 72 degrees)
        segment_angle = np.mod(np.arctan2(y, x), np.pi / 2.5)
        if segment_angle < np.pi / 5:
            return radius_outer
        else:
            return radius_inner



    def Get_Data(self,**kwagrs)->dde.data.PDE: #u(x,t)
        
        # 创建星形域
        if self.shape == "square":
            self.geom =  dde.geometry.Rectangle([-1, -1], [1, 1])
        elif self.shape == "poly_with_hole":
            vertices = np.array([
                [0, 0],   # 左下角
                [1, 0],   # 右下角
                [1, 0.5], # 右侧中间凸起
                [1.5, 1], # 右上角凸起
                [1, 1.5], # 右侧中间凹槽
                [1, 2],   # 右上角
                [0, 2],   # 左上角
                [-0.5, 1] # 左侧中间凹槽
            ])
            # 定义圆形洞的中心和半径
            hole_center = [0.5, 1]  # 假设洞位于多边形的中心
            hole_radius = 0.3  # 圆形洞的半径

            # 使用Circle类创建圆形洞
            circle_hole = dde.geometry.geometry_2d.Disk(hole_center, hole_radius)
            # 使用Polygon类创建复杂多边形形状
            complex_polygon = dde.geometry.Polygon(vertices)
            # 使用CSGDifference从复杂多边形中减去圆形洞，创建一个新的几何形状
            complex_polygon_with_hole = dde.geometry.CSGDifference(complex_polygon, circle_hole)
            self.geom = complex_polygon_with_hole
        elif self.shape == "Circle":
            # 定义圆形洞的中心和半径
            hole_center = [0, 0]  
            hole_radius = 1 # 圆形洞的半径

            # 使用Circle类创建圆形洞
            circle_hole = dde.geometry.geometry_2d.Disk(hole_center, hole_radius)
            self.geom = circle_hole
        elif self.shape =="Triangle":
            # 定义三角形的三个顶点

            # 使用Polygon类创建三角形形状
            triangle = dde.geometry.geometry_2d.Triangle([-1, -1], [1, -1], [0, 1])
            self.geom = triangle
        
        domain_numbers=kwagrs.get("domain_numbers",6400)
        boundary_numbers=kwagrs.get("boundary_numbers",2000)
        test_numbers=kwagrs.get("test_numbers",6400)

        
        
        bc = dde.icbc.DirichletBC(self.geom, lambda x: 0, self.boundary)
        self.data = dde.data.PDE(
            self.geom,
            self.pde,
            bc,
            num_domain=domain_numbers,#sqrt(3600)=60
            num_boundary=boundary_numbers,
            num_test=test_numbers,
            train_distribution="pseudo",
        )

        return self.data

    def possion_eq_exact_solution(self,x, y):
        """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

        Parameters
        ----------
        x : np.ndarray
        y : np.ndarray
        """
    
        usol = self.np_u(x, y)
        # # 检查 x 和 y 是否靠近边界 (例如，靠近 0 或 1)
        # for i in range (len(x)):
        #      if self.geom.on_boundary((x[i], y[i])):
        #         usol[i] = np.sin(self.mu * x[i]**2) + np.sin(self.mu* y[i]**2).reshape(-1,1)
        return usol

    def gen_exact_solution(self,x,y):
        """Generates exact solution for the heat equation for the given values of x and t."""

        usol= self.possion_eq_exact_solution(x, y)
    
        return usol

    def plot_exact( self,
                    ax=None,
                    title="Exact u(x,y)", 
                    cmap="bwr",data=None):
        x = data[:, 0]
        y = data[:, 1]

        u=self.gen_exact_solution(x,y)

        sc=ax.scatter(x,y, c=u,cmap="bwr",s=2)

        ax.set_xlabel("x",fontsize=12)
        
        ax.set_ylabel("y",fontsize=12)
           
        # 色条
        cb=plt.colorbar(sc, ax=ax)

        ax.set_title(title,fontsize=14)
        return u,cb
    def plot_pred(self,ax=None,model=None,
                    title="Pred u(x,y)",
                    cmap="bwr",data=None):
        # Number of points in each dimension:
        # 提取 x 和 t
        x = data[:, 0]
        y = data[:, 1]
        
        data=torch.from_numpy(data).float().to(self.device)
    
        # 获取 usol 值
        if self.device =="cuda":
               
            usol_net=model(data).cpu().detach().numpy()
          
        else:
            usol_net = model(data).cpu().detach().numpy()
        

        # 绘制热力图
        print(usol_net.shape)
        sc=ax.scatter(x,y, c=usol_net, cmap="bwr",s=2)
        cb= plt.colorbar(sc, ax=ax)

        ax.set_xlabel("x",fontsize=12)
        ax.set_ylabel("y",fontsize=12)
        ax.set_title(title,fontsize=14)
        return usol_net,cb

    def train(self,net=None):
        pass

# Redefine the functions to accept a point (x, y) instead of a vector x
def f_point(x, y, mu=15):
    return 4 * mu**2 * x**2 * np.sin(mu * x**2) - 2 * mu * np.cos(mu * x**2) + \
           4 * mu**2 * y**2 * np.sin(mu * y**2) - 2 * mu * np.cos(mu * y**2)

def u_point(x, y, mu=15):
    return np.sin(mu * x**2) + np.sin(mu * y**2)




if __name__ == "__main__":
    po = PDE_PossionData()

    
    # Generate 1024 points in 2D space
    points = np.random.rand(10240, 2) # This will create points with coordinates between 0 and 1

    # Compute f(x) and u(x) for each point
    f_values = np.array([f_point(x,y) for x,y in points])
    u_values = np.array([u_point(x,y) for x,y in points])

    import matplotlib.pyplot as plt

    # Split the points into x and y coordinates for plotting
    x_coords, y_coords = points.T

    # Create a scatter plot for the values of f(x, y)
    plt.figure(figsize=(14, 7))

    # Left plot for f(x, y)
    plt.subplot(1, 2, 1)
    plt.scatter(x_coords, y_coords, c=f_values, cmap='viridis')
    plt.colorbar(label='f(x, y) value')
    plt.title('f(x, y) over 1024 points')
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')

    
    # plt.ylabel('y coordinate')  # Not needed on the right plot, shared y
    plt.show()








