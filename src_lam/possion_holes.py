import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import TensorDataset
class Produce_Data_set():  # 训练集，验证集，测试集 7:2:1
    def __init__(self, name="holes"):
        self._name=name

    def produce_data(self, path,tensor):
        self._num=tensor.shape[0]
        # 文件夹
        if not os.path.exists(path + f"{self._name}"):
            os.makedirs(path + f"{self._name}")

        tensor = torch.from_numpy(tensor).to(torch.float32)
        # 划分数据集：训练集，验证集，测试集为 7:2:1
        train_size = int(0.7 * self._num)
        val_size = int(0.2 * self._num)

        train_data=TensorDataset(tensor[:train_size,:,0:2],
                                 tensor[:train_size,:,2:3])

        val_data = TensorDataset(tensor[train_size:train_size + val_size,:,0:2],
                                 tensor[train_size:train_size + val_size,:,2:3]) #input,label
        test_data = TensorDataset(tensor[train_size + val_size:,:,0:2],
                                  tensor[train_size + val_size:,:,2:3])


        # 保存
        torch.save(train_data, path + f"{self._name}"+ '/train_loader.pt')
        torch.save(val_data, path + f"{self._name}" + '/val_loader.pt')
        torch.save(test_data, path + f"{self._name}" + '/test_loader.pt')
class PoissonEquationWithHoles:
    def __init__(self,
                 domain_size,
                 circle_params,
                 ellipse_params,
                 num_samples_domain,
                 num_samples_circles,
                 num_samples_ellipse):
        """
        初始化函数
        :param domain_size: 域边界的尺寸
        :param circle_params: 圆的参数列表，每个元素是一个元组(a, b, r)，分别代表圆心和半径
        :param ellipse_params: 椭圆的参数列表，每个元素是一个元组(h, k, a, b)，分别代表中心和半轴
        :param num_samples_domain: 域边界的采样数量
        :param num_samples_circles: 三个圆的采样数量列表
        :param num_samples_ellipse: 椭圆的采样数量
        """

        self.domain_size = domain_size
        self.circle_params = circle_params
        self.ellipse_params = ellipse_params
        self.num_samples_domain = num_samples_domain
        self.num_samples_circles = num_samples_circles
        self.num_samples_ellipse = num_samples_ellipse

    def sample_domain_boundary(self):
        # 随机生成边界点
        points = []
        while len(points) < self.num_samples_domain:
            x = np.random.uniform(-self.domain_size, self.domain_size)
            y = np.random.uniform(-self.domain_size, self.domain_size)

            # Check if the point is outside the circles and the ellipse
            if not any((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2 for cx, cy, r in self.circle_params):
                if ((x - self.ellipse_params[0][0]) ** 2 / self.ellipse_params[0][2] ** 2 +
                        (y - self.ellipse_params[0][1]) ** 2 / self.ellipse_params[0][3] ** 2 > 1):
                    points.append((x, y))

        # Convert list of points to NumPy arrays
        x_samples, y_samples = zip(*points)  # Unzipping the list of tuples
        return np.array(x_samples), np.array(y_samples)
    def sample_circle_boundary(self, a, b, r, num_samples):
        x_samples = a + np.random.uniform(-r, r, num_samples)
        y_samples = b + np.random.choice([-1, 1], num_samples) * np.sqrt(r ** 2 - (x_samples - a) ** 2)
        return x_samples, y_samples
    def sample_ellipse_boundary(self, h, k, a, b, num_samples):
        x_samples = h + np.random.uniform(-a, a, num_samples)
        y_samples = k + np.random.choice([-1, 1], num_samples) * b * np.sqrt(1 - ((x_samples - h) ** 2 / a ** 2))
        return x_samples, y_samples
    def sample_visualize(self):
        plt.figure(figsize=(8, 8))
        # 域边界采样点
        x_domain, y_domain = self.sample_domain_boundary()
        plt.scatter(x_domain, y_domain, label='Domain Boundary', color='black')

        # 圆的采样点
        for i, (a, b, r) in enumerate(self.circle_params):
            x_circle, y_circle = self.sample_circle_boundary(a, b, r, self.num_samples_circles[i])
            plt.scatter(x_circle, y_circle, label=f'Circle center=({a},{b}), r={r}')
        # 椭圆的采样点
        for h, k, a, b in self.ellipse_params:
            x_ellipse, y_ellipse = self.sample_ellipse_boundary(h, k, a, b, self.num_samples_ellipse)
            plt.scatter(x_ellipse, y_ellipse, label=f'Ellipse center=({h},{k}), a={a}, b={b}')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Sampled Boundary Points for Domain, Circles, and Ellipse')
        plt.legend(loc='upper right')
        plt.show()
    def exact_solution(self, x1, x2, mu):

        return np.sin(mu * x1) * np.sin(mu * x2)
    def plot_contour(self, mu,**kwargs):
        # Generate grid for the domain
        x = np.linspace(-self.domain_size, self.domain_size, 1000)
        y = np.linspace(-self.domain_size, self.domain_size, 1000)
        X, Y = np.meshgrid(x, y)

        # Compute the solution for the entire domain
        U = self.exact_solution(X, Y, mu)

        # Mask the holes (circles and ellipse)
        for (center_x, center_y, radius) in self.circle_params:
            mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
            U[mask] = 0  # Assign NaN to holes

        # Mask the ellipse
        h, k, a, b = self.ellipse_params[0]
        ellipse_mask = (X - h)**2 / a**2 + (Y - k)**2 / b**2 <= 1
        U[ellipse_mask] = 0  # Assign NaN to the elliptic hole

        # Plot the contour of the solution, avoiding the holes
        plt.figure(figsize=(8, 8))
        contour = plt.contourf(X, Y, U, levels=100, cmap='bwr')
        plt.colorbar(contour)
        plt.title('Plot of the Exact Solution')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
    def sample_exact_solution(self, mu=7*np.pi):
        # Sample points from the domain boundary
        x_domain, y_domain = self.sample_domain_boundary()
        u_domain = self.exact_solution(x_domain, y_domain, mu)

        # Combine domain boundary points into one array
        domain_points_values = np.column_stack((x_domain, y_domain, u_domain))

        # Initialize an array to hold all points and values
        all_points_values = domain_points_values

        # Sample and combine points from the circle boundaries
        for i, (a, b, r) in enumerate(self.circle_params):
            x_circle, y_circle = self.sample_circle_boundary(a, b, r, self.num_samples_circles[i])
            u_circle = np.zeros_like(x_circle)
            circle_points_values = np.column_stack((x_circle, y_circle, u_circle))
            all_points_values = np.vstack((all_points_values, circle_points_values))

        # Sample and combine points from the ellipse boundary
        h, k, a, b = self.ellipse_params[0]
        x_ellipse, y_ellipse = self.sample_ellipse_boundary(h, k, a, b, self.num_samples_ellipse)
        u_ellipse = np.zeros_like(x_ellipse)
        ellipse_points_values = np.column_stack((x_ellipse, y_ellipse, u_ellipse))
        all_points_values = np.vstack((all_points_values, ellipse_points_values))
        #5000,3
        return all_points_values
    def sample(self,batch):
        '''
        return [batch,5000,3]
        Returns:
        '''
        all_points_values_all=None
        for j in range(batch):
            all_points_values=self.sample_exact_solution(mu=7*np.pi)
            all_points_values=all_points_values[np.newaxis,:,:]
            try:
                all_points_values_all=np.concatenate((all_points_values_all,all_points_values),axis=0)
            except:
                all_points_values_all=all_points_values


        return all_points_values_all
    def plot_2dfrom_model(self,model,ax,title,cmap='bwr'):
        '''
        plot 2d figure from model
        Args:
            model: from AI model

        Returns: ax

        '''
        # Generate grid for the domain
        x = np.linspace(-self.domain_size, self.domain_size, 200)
        y = np.linspace(-self.domain_size, self.domain_size, 200)
        X, Y = np.meshgrid(x, y)

        # Compute the solution for the entire domain
        if model is None:
            U = self.exact_solution(X, Y, mu=7*np.pi)
        else:
            device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            XY_combined = np.dstack((X, Y))
            # 重塑数组为 (10000, 2)
            XY_reshaped = XY_combined.reshape(-1, 2)
            XY_reshaped = torch.from_numpy(XY_reshaped).float().to(device)
            U = model(XY_reshaped).cpu().detach().numpy()
            U= U.reshape(200, 200)
        # Mask the holes (circles and ellipse)
        for (center_x, center_y, radius) in self.circle_params:
            mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
            U[mask] = np.nan # Assign NaN to holes
        # Mask the ellipse
        h, k, a, b = self.ellipse_params[0]
        ellipse_mask = (X - h) ** 2 / a ** 2 + (Y - k) ** 2 / b ** 2 <= 1
        U[ellipse_mask] = np.nan  # Assign NaN to the elliptic hole
        # Plot the contour of the solution, avoiding the holes
        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.contourf(X, Y, U, levels=200, cmap=cmap)
        return U

if __name__=="__main__":

    # 使用示例
    domain_size = 1.0
    circle_params = [(-0.6, -0.6, 0.3), (0.3, -0.3, 0.6), (0.6, 0.6, 0.3)]  # 圆的参数
    ellipse_params = [(-0.5, 0.5, 1 / 4, 1 / 8)]  # 椭圆的参数
    num_samples_domain = 2400#包括边界
    num_samples_circles = [550, 1100, 550]  # 三个圆的采样数量
    num_samples_ellipse = 400

    poisson = PoissonEquationWithHoles(domain_size,
                                       circle_params,
                                       ellipse_params,
                                       num_samples_domain,
                                       num_samples_circles,
                                       num_samples_ellipse)
    poisson.sample_visualize()
    poisson.plot_contour(mu=7*np.pi)
    #5000,3
    sampled_points_values = poisson.sample_exact_solution(mu=7*np.pi)
    sample=poisson.sample(100) #采样1000次 [1000,5000,3]
    dataset=Produce_Data_set()#生成数据集
    dataset.produce_data(path="../data/possion_2d/",tensor=sample)


    # plt.scatter(sampled_points_values[:, 0], sampled_points_values[:, 1],
    #             c=sampled_points_values[:, 2])
    # plt.show()
    val=torch.load("../data/possion_2d/holes/val_loader.pt")
    # 例子：检查 DataLoader 是否正确加载
    for inputs, labels in val:
        print(inputs.shape, labels.shape)
        break  # 只打印第一批数据，然后跳出循环