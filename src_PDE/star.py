
import numpy as np
#rho_theta :star-shaped domain; U(x,y) :exact solution

def rho_theta(theta):
    return 1+np.cos(4*theta)**2
    
def rho_xy(x,y):
    return rho_theta(np.angle(x+y*1j))

def x_theta(theta):
    return rho_theta(theta)*np.cos(theta)

def y_theta(theta):
    return rho_theta(theta)*np.sin(theta)

def R(x,y):
    return np.sqrt(x**2+y**2)

k=100
def U(x,y):
    return np.cos(x+k*x*y+k*y**2)
 
 
delta = 0.01
x_domain = np.arange(-2, 2, delta)
y_domain = np.arange(-2, 2, delta)
x_mesh,y_mesh = np.meshgrid(x_domain, y_domain)


governing_equation_components = []
governing_equation_components.append(lambda x: x[3])
governing_equation_components.append(lambda x: x[2]*x[4])


governing_equation_mask = R(x_mesh,y_mesh)<=rho_xy(x_mesh,y_mesh)

fx = governing_equation_mask*(
    (-np.sin(k*y_mesh**2+k*x_mesh*y_mesh+x_mesh)*(k*y_mesh+1)) + 
    (np.cos(k*y_mesh**2+k*x_mesh*y_mesh+x_mesh))*(-np.sin(k*y_mesh**2+k*x_mesh*y_mesh+x_mesh)*(k*x_mesh+2*k*y_mesh)))

observation_components = []
observation_components.append(lambda x: x[2])
observation_components.append(lambda x: x[3])
observation_components.append(lambda x: x[4])

observation_data = []

theta_list = np.linspace(0,np.pi*2,800)
for theta in theta_list:
    x = x_theta(theta)
    y = y_theta(theta)
    comb = [1,0,0]
    v = U(x,y)
    observation_data.append([x,y,comb,v])
    
import matplotlib.pyplot as plt

# 创建掩码
governing_equation_mask = R(x_mesh, y_mesh) <= rho_xy(x_mesh, y_mesh)

# 计算U在整个网格上的值
U_values = U(x_mesh, y_mesh)

# 使用掩码将U的值限制在星形域内
U_masked = np.ma.array(U_values, mask=~governing_equation_mask)

# 绘图
plt.figure(figsize=(8, 8))
plt.pcolormesh(x_mesh, y_mesh, U_masked, shading='auto', cmap='viridis')  # 正确传递U_masked
plt.colorbar()  # 显示颜色条
plt.xlabel('x')
plt.ylabel('y')
plt.title('U(x,y) in a star-shaped domain')
plt.savefig("star.png")
plt.show()


#