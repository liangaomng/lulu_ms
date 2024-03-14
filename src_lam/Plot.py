import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from .Analyzer import Analyzer4scale
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
# 根据行列数自动创建子图
class Plot_Adaptive:
    def __init__(self):
        self.fig = None
        self.axes = None
    def _create_subplot_grid1(self,nrow, ncol):
        self.fig = plt.figure(figsize=(1.6*ncol * 3, 1.4*nrow * 3))
        gs = GridSpec(nrow, ncol, figure=self.fig,hspace=0.4,wspace=0.3)
        self.axes = []

        # 添加跨列的子图
           # 第一行的图
        for c in range(ncol):
            ax = fig.add_subplot(gs[0, c])
            axes.append(ax)
        # 添加第2行的图，使用整个列跨度，调整高度
        ax = fig.add_subplot(gs[1, :])
        axes.append(ax)

        if ncol >= 2:
            # 第三行的两个子图，根据需要调整跨度和位置
            ax1 = fig.add_subplot(gs[-1, :ncol//2])  # 假设第一个子图占据左半边
            ax2 = fig.add_subplot(gs[-1, ncol//2:])  # 假设第二个子图占据右半边
            axes.extend([ax1, ax2])
                
         
            self.axes.append(ax)
          

    def _create_subplot_grid2(self,nrow, ncol):
        self.fig = plt.figure(figsize=(2.2 * ncol * 3, 1.8 * nrow * 3))
        gs = GridSpec(nrow, ncol, figure=self.fig, hspace=0.4, wspace=0.3)
        self.axes = []

        # 添加第一行的子图，每列一个
        for c in range(ncol):
            ax = self.fig.add_subplot(gs[0, c])
            self.axes.append(ax)

        # 添加第二行的子图，使用整个列跨度
        ax = self.fig.add_subplot(gs[1, :])
        self.axes.append(ax)

        # 确保第三行的子图大小一致
        if ncol >= 2:
            # 第三行的两个子图，平均分配列跨度
            mid_point = ncol // 2 if ncol % 2 == 0 else ncol // 2 + 1
            ax1 = self.fig.add_subplot(gs[2:4, :mid_point])  # 第一个子图占据左半边
            ax2 = self.fig.add_subplot(gs[2:4, mid_point:])  # 第二个子图占据右半边
            self.axes.extend([ax1, ax2])


    def _create_subplot_moe_grid2(self,nrow,ncol):
        self.fig = plt.figure(figsize=(2.2 * ncol * 3, 1.8 * nrow * 3)) 
        gs = GridSpec(nrow, ncol, figure=self.fig, hspace=0.4, wspace=0.3)
        self.axes = []
        # 添加第一行的子图，每列一个
        for c in range(ncol):
            ax = self.fig.add_subplot(gs[0, c])
            self.axes.append(ax)
        # 添加第二行的子图，使用整个列跨度
        ax = self.fig.add_subplot(gs[1, :])
        self.axes.append(ax)
        # 添加第一行的子图，每列一个
        for c in range(ncol):
            ax = self.fig.add_subplot(gs[2, c])
            self.axes.append(ax)
                # 添加第一行的子图，每列一个
        for c in range(ncol):
            ax = self.fig.add_subplot(gs[3, c])
            self.axes.append(ax)
                # 添加第一行的子图，每列一个
        for c in range(ncol):
            ax = self.fig.add_subplot(gs[4, c])
            self.axes.append(ax)
        for c in range(ncol):
            ax = self.fig.add_subplot(gs[5, c])
            self.axes.append(ax)




    def plot_moe__load_2d(self, nrow,ncol,**kwagrs):
        
        if self.fig is None:
            self._create_subplot_moe_grid2(nrow=6, ncol=3)

        load = kwagrs['load']
        # 计算每个点的y值（即每个数组的长度）
        y_values = [arr.shape[0] for arr in load]
     
      
        #画图计算的solver
        solver=kwagrs["solver"]
        model=kwagrs["model"]
        omega_value= kwagrs["omega_record"]
                #负载
        cmap = plt.cm.rainbow
        norm = plt.Normalize(min(y_values), max(y_values))
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        

        # 注意：这里我们使用了matplotlib的颜色映射。更亮的颜色表示更大的y值。

        for i, ax in enumerate(self.axes):
            
            if i == 2:
                epoch = kwagrs["epoch"]

                # 在第一个子图上绘制真实值的散点图

                U_true,cb1=solver.plot_exact(   ax=self.axes[0],
                                                title="True",
                                                cmap="bwr",
                                                data=solver.data.test_x)

                # 在第一个子图上绘制真实值的散点图
                U_pred,cb2=solver.plot_pred(    ax=self.axes[1],
                                                model=model,
                                                title="Pred",
                                                cmap="bwr",
                                                data=solver.data.test_x,MOE=True)
                U_true=U_true.reshape(-1,1)
                
                data=solver.data.test_x
                
                #x轴实际是t【:,1】
                sc=self.axes[2].scatter(data[:,0],data[:,1],c=np.abs(U_pred-U_true),cmap="bwr",s=2)
                # 添加颜色条
                cb3=plt.colorbar(sc, ax=self.axes[2], label='Absolute Difference')

                # 隐藏 x 和 y 轴的刻度和标签
                self.axes[2].set_xticks([])
                self.axes[2].set_yticks([])
                # 设置第一个子图的图例、坐标轴标签和标题
                mse = np.mean((U_pred - U_true) ** 2)
                self.axes[2].set_title("MSE={:.6f}_Epoch{}".format(mse, epoch),fontsize=14) 
            if i == 3:  # 第二张图
                # 在第最后子图上绘制损失曲线
            
                loss_record_df = kwagrs["loss_record"]

                ax.plot(loss_record_df[:,0], loss_record_df[:,1], label="Train Loss", color="blue")
                #ax.plot(loss_record_df[:,0], loss_record_df[:,2], label="Valid Loss", color="red")
                ax.plot(loss_record_df[:,0], loss_record_df[:,2], label="Test Loss", color="green",alpha=0.8,marker=">")
                
                ax.plot(loss_record_df[:,0], loss_record_df[:,3], label="PDE Loss", color="purple",alpha=0.8,marker="*")
                ax.plot(loss_record_df[:,0], loss_record_df[:,4], label="Boundary Loss", color="black",marker="+")
                ax.plot(loss_record_df[:,0], loss_record_df[:,5], label="Data Loss", color="#FF4500",marker="o")
            
                # # 画三条虚线 得到最小损失的点
                if epoch >= 1:
                    # for j, value in enumerate(Record):
                    #     ax.axvline(x=value, color=c_map[j], linestyle='--')
                    for loss_type in ['Data loss']: #test 只关注data loss
                        # data loss   在序号为5的列
                        min_loss = np.min(loss_record_df[:, 5])
                        min_epoch = loss_record_df[np.where(loss_record_df[:, 5] == min_loss)][0, 0]
                        ax.axhline(y=min_loss, xmax=min_epoch, color='black', linestyle=':', linewidth=4)
                        # 在最小损失点做标记
                        ax.plot(min_epoch, min_loss, '*',
                                color='black', markersize=18,
                                label=f"Test_Data_Min{min_loss:.1e}")  # 使用黑色圆点做标记
                        # 假设已经计算出min_loss，将其添加到Y轴的刻度标签中
                        extra_ticks = ax.get_yticks().tolist() + [min_loss]
                        ax.set_yticks(extra_ticks)
                        # 设置刻度标签，确保最小损失值的标签使用科学记数法
                        ax.axvline(x=min_epoch, ymin=1e-7, ymax=min_loss, linestyle='--', linewidth=4)
                    for loss_type in ['PDE loss']: #test 只关注pde loss
                        # data loss   在序号为5的列
                        min_loss = np.min(loss_record_df[:, 3])
                        min_epoch = loss_record_df[np.where(loss_record_df[:, 3] == min_loss)][0, 0]
                        ax.axhline(y=min_loss, xmax=min_epoch, color='black', linestyle=':', linewidth=4)
                        # 在最小损失点做标记
                        ax.plot(min_epoch, min_loss, 's',
                                color='black', markersize=15,
                                label=f"Test_PDE_Min{min_loss:.1e}")  # 使用黑色圆点做标记
                        # 假设已经计算出min_loss，将其添加到Y轴的刻度标签中
                        extra_ticks = ax.get_yticks().tolist() + [min_loss]
                        ax.set_yticks(extra_ticks)
                        # 设置刻度标签，确保最小损失值的标签使用科学记数法
                        ax.axvline(x=min_epoch, ymin=1e-7, ymax=min_loss, linestyle='--', linewidth=4)
                ax.legend(loc="best", fontsize=12)
                # 设置第二个子图的图例、坐标轴标签和标题
                ax.set_yscale('log')  # 将y轴设置为对数尺度
                ax.set_xlabel('Epoch', fontsize=16)
                ax.set_ylabel('Loss', fontsize=16)
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                ax.tick_params(labelsize=13, width=2, colors='black')
                ax.set_title("Loss_Epoch{}".format(epoch))

            if i == 12:# 散点
            #正常画9个子网络的load
                ax_order=4
                
                for j, y in enumerate(y_values):


                    coord_x=load[j][:,0].cpu().detach().numpy()

                    coord_y=load[j][:,1].cpu().detach().numpy()

                    self.axes[ax_order].scatter(coord_x,coord_y,alpha=0.8,s=10,label=f"subnet_{j}_load")
                    self.axes[ax_order].legend(loc="upper left")
                    self.axes[ax_order].set_title(f"subnet_{j}_load_distribution_{y_values[j]}")
                    ax_order = ax_order+1

                

            if i ==13:#  pde gates
                pde_gates = kwagrs["p_gates"]
                im=ax.imshow(pde_gates, cmap='bwr',aspect="auto",vmin=0,vmax=0.5)
                ax.set_title("p_gates",fontsize=18 )
            
                ax.set_xlabel('subnets',fontsize=15)
                # 在特定的列上绘制虚线
                num_columns = pde_gates.shape[1]
                for col in range(num_columns):
                    ax.axvline(x=col, color='k', linestyle='--', linewidth=1)
                cb4 = plt.colorbar(im, ax=self.axes[13], label='Prob compare with experts')
                # 设置横坐标刻度的字体大小
   
                 

            if i ==14:
                bc_gates = kwagrs["b_gates"]
                im = ax.imshow(bc_gates, cmap='bwr',aspect="auto",vmin=0,vmax=0.5)
                ax.set_title("bc_gates",fontsize=18)
                ax.set_xlabel('subnets',fontsize=15)
                for col in range(num_columns):
                    ax.axvline(x=col, color='k', linestyle='--', linewidth=1)
                cb5 = plt.colorbar(im, ax=ax, label='Prob compare with experts')
            

            if i == 15:
            

                # 绘制散点图，其中颜色亮度基于y值
                for j, y in enumerate(y_values):
                    color = scalar_map.to_rgba(y)
                    if j == 0:
                        ax.scatter(x=j, y=y, color=color,alpha=0.8,s=100,label="load")
                    else:
                        ax.scatter(x=j, y=y, color=color,s=100,alpha=0.8)
                ax.plot(range(len(y_values)), y_values, '--', color='k')
                cb6=plt.colorbar(scalar_map, ax=ax, label='Load in test')


        return self.fig, self.axes,[cb1,cb2,cb3,cb4,cb5,cb6]
        



            
    def plot_1d(self,nrow,ncol,**kwagrs):
        # 绘制一些示例数据
        c_map=["Green","Blue","Purple"]
        # 从kwagrs中获取参数
        analyzer=kwagrs["analyzer"]
        if self.fig is None:
            self._create_subplot_grid1(nrow,ncol)
        Record=kwagrs["contribution_record"]
        loss_record_df = kwagrs["loss_record"]
        x_test = kwagrs["x_test"]
        pred = kwagrs["pred"]
        y_true = kwagrs["y_true"]
        avg_test_loss = kwagrs["avg_test_loss"]
        epoch = kwagrs["epoch"]

        for i, ax in enumerate(self.axes):

            if i==0: #   第一张图
                # 在第一个子图上绘制预测值的散点图
                ax.cla()

                ax.scatter(x_test, pred, label="Pred", color="red")
                # 在第一个子图上绘制真实值的散点图
                ax.scatter(x_test, y_true, label="True", color="blue")
                # 设置第一个子图的图例、坐标轴标签和标题
                ax.legend(loc="best", fontsize=16)
                ax.set_xlabel('x', fontsize=16)
                ax.set_ylabel('y', fontsize=16)
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                ax.tick_params(labelsize=16, width=2, colors='black')
                ax.set_title("Test_MSE={:.6f}_Epoch{}".format(avg_test_loss, epoch))
                ax.legend()
            if i==1: #   第二张图
                # 在第最后子图上绘制损失曲线
                ax.cla()
                ax.plot(loss_record_df[:,0], loss_record_df[:,1], label="Train Loss", color="blue")
                ax.plot(loss_record_df[:,0], loss_record_df[:,2], label="Valid Loss", color="red")
                ax.plot(loss_record_df[:,0], loss_record_df[:,3], label="Test Loss", color="green")

                # 画三条虚线
                if epoch >=100:
                    for j,value in enumerate(Record):
                        ax.axvline(x=value, color=c_map[j], linestyle='--')
                    for loss_type in ['Test Loss']:
                        #valid loss
                        min_loss = np.min(loss_record_df[:,2])
                        min_epoch = loss_record_df[np.where(loss_record_df[:, 2] == min_loss)][0, 0]
                        ax.axhline(y=min_loss, xmax=min_epoch,color='black', linestyle=':',linewidth=4)
                        # 在最小损失点做标记
                        ax.plot(min_epoch,min_loss, '*',
                                color='black',markersize=18,
                                label=f"valid_{min_loss:.1e}")  # 使用黑色圆点做标记
                        # 假设已经计算出min_loss，将其添加到Y轴的刻度标签中
                        extra_ticks = ax.get_yticks().tolist() + [min_loss]
                        ax.set_yticks(extra_ticks)
                        # 设置刻度标签，确保最小损失值的标签使用科学记数法
                        ax.axvline(x=min_epoch,ymin=1e-10,ymax=min_loss ,linestyle='--',linewidth=4)
                ax.legend(loc="best", fontsize=16)
                # 设置第二个子图的图例、坐标轴标签和标题
                ax.set_yscale('log')  # 将y轴设置为对数尺度
                ax.set_xlabel('Epoch', fontsize=16)
                ax.set_ylabel('Loss', fontsize=16)
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                ax.tick_params(labelsize=16, width=2, colors='black')
                ax.set_title("Loss_Epoch{}".format(epoch))

            if i==2: #   第三行图开始画贡献度
                if (epoch == Record[0]):
                    analyzer.plot_contributions(ax=self.axes[i],fig=self.fig,cmap=c_map[0])
            if i==3:
                if (epoch == Record[1]):
                    analyzer.plot_contributions(ax=self.axes[i],fig=self.fig,cmap=c_map[1])
            if i==4:
                if (epoch == Record[2]):
                    # for j in epoch_axv:
                    analyzer.plot_contributions(ax=self.axes[i],fig=self.fig,cmap=c_map[2])


        return self.fig,self.axes
    def plot_2d(self, nrow, ncol, **kwagrs):
        # 绘制一些示例数据
        c_map = ["Green", "Blue", "Purple"]
        # 从kwagrs中获取参数
      
        #画图计算的solver
        solver=kwagrs["solver"]
        model=kwagrs["model"]
        if self.fig is None:
            self._create_subplot_grid2(nrow, ncol)
        contri_value = kwagrs["contr_record"]
        omega_value= kwagrs["omega_record"]

      

        for i, ax in enumerate(self.axes):
        
            if i == 2:  # 第一张图
                # 在第一个子图上绘制预测值的散点图

                epoch = kwagrs["epoch"]
                # 在第一个子图上绘制真实值的散点图
                U_true,cb1=solver.plot_exact(   ax=self.axes[0],
                                                title="True",
                                                cmap="bwr",
                                                data=solver.data.test_x)

                # 在第一个子图上绘制真实值的散点图
                U_pred,cb2=solver.plot_pred(    ax=self.axes[1],
                                                model=model,
                                                title="Pred",
                                                cmap="bwr",
                                                data=solver.data.test_x,
                                                MOE=False)
                U_true=U_true.reshape(-1,1)
                
                data=solver.data.test_x
                
                #x轴实际是t【:,1】
                sc=self.axes[2].scatter(data[:,0],data[:,1],c=np.abs(U_pred-U_true),cmap="bwr",s=2)
                # 添加颜色条
                cb3=plt.colorbar(sc, ax=self.axes[2], label='Absolute Difference')

                # 隐藏 x 和 y 轴的刻度和标签
                self.axes[2].set_xticks([])
                self.axes[2].set_yticks([])
                # 设置第一个子图的图例、坐标轴标签和标题
                mse = np.mean((U_pred - U_true) ** 2)
                self.axes[2].set_title("MSE={:.6f}_Epoch{}".format(mse, epoch),fontsize=14)


            if i == 3:  # 第二张图
                # 在第最后子图上绘制损失曲线
            
                loss_record_df = kwagrs["loss_record"]

                ax.plot(loss_record_df[:,0], loss_record_df[:,1], label="Train Loss", color="blue")
                #ax.plot(loss_record_df[:,0], loss_record_df[:,2], label="Valid Loss", color="red")
                ax.plot(loss_record_df[:,0], loss_record_df[:,2], label="Test Loss", color="green",alpha=0.8,marker=">")
                
                ax.plot(loss_record_df[:,0], loss_record_df[:,3], label="PDE Loss", color="purple",alpha=0.8,marker="*")
                ax.plot(loss_record_df[:,0], loss_record_df[:,4], label="Boundary Loss", color="black",marker="+")
                ax.plot(loss_record_df[:,0], loss_record_df[:,5], label="Data Loss", color="#FF4500",marker="o")
            
                # # 画三条虚线 得到最小损失的点
                if epoch >= 1:
                    # for j, value in enumerate(Record):
                    #     ax.axvline(x=value, color=c_map[j], linestyle='--')
                    for loss_type in ['Data loss']: #test 只关注data loss
                        # data loss   在序号为5的列
                        min_loss = np.min(loss_record_df[:, 5])
                        min_epoch = loss_record_df[np.where(loss_record_df[:, 5] == min_loss)][0, 0]
                        ax.axhline(y=min_loss, xmax=min_epoch, color='black', linestyle=':', linewidth=4)
                        # 在最小损失点做标记
                        ax.plot(min_epoch, min_loss, '*',
                                color='black', markersize=18,
                                label=f"Test_Data_Min{min_loss:.1e}")  # 使用黑色圆点做标记
                        # 假设已经计算出min_loss，将其添加到Y轴的刻度标签中
                        extra_ticks = ax.get_yticks().tolist() + [min_loss]
                        ax.set_yticks(extra_ticks)
                        # 设置刻度标签，确保最小损失值的标签使用科学记数法
                        ax.axvline(x=min_epoch, ymin=1e-7, ymax=min_loss, linestyle='--', linewidth=4)
                    for loss_type in ['PDE loss']: #test 只关注pde loss
                        # data loss   在序号为5的列
                        min_loss = np.min(loss_record_df[:, 3])
                        min_epoch = loss_record_df[np.where(loss_record_df[:, 3] == min_loss)][0, 0]
                        ax.axhline(y=min_loss, xmax=min_epoch, color='black', linestyle=':', linewidth=4)
                        # 在最小损失点做标记
                        ax.plot(min_epoch, min_loss, 's',
                                color='black', markersize=15,
                                label=f"Test_PDE_Min{min_loss:.1e}")  # 使用黑色圆点做标记
                        # 假设已经计算出min_loss，将其添加到Y轴的刻度标签中
                        extra_ticks = ax.get_yticks().tolist() + [min_loss]
                        ax.set_yticks(extra_ticks)
                        # 设置刻度标签，确保最小损失值的标签使用科学记数法
                        ax.axvline(x=min_epoch, ymin=1e-7, ymax=min_loss, linestyle='--', linewidth=4)
                ax.legend(loc="best", fontsize=12)
                # 设置第二个子图的图例、坐标轴标签和标题
                ax.set_yscale('log')  # 将y轴设置为对数尺度
                ax.set_xlabel('Epoch', fontsize=16)
                ax.set_ylabel('Loss', fontsize=16)
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
       
                ax.set_title("Loss_Epoch{}".format(epoch))
            if i == 4:  # 保存
                # 归一化contri值 因为【epoch,9】，对每一个epoch做归一化 对每一行
            

                epsilon = 1e-10  # 一个非常小的正数，防止分母为零

                min_values = np.amin(contri_value[:,1:], axis=1, keepdims=True)
                max_values = np.amax(contri_value[:,1:], axis=1, keepdims=True)

                range_values = max_values - min_values + epsilon  # 加上 epsilon 避免分母为零
                contri_normalized = (contri_value[:,1:] - min_values) / range_values

             
                cb4 = ax.matshow(contri_normalized.T,cmap='bwr', aspect='auto', vmin=0, vmax=0.3)

                # 设置轴标签和标题
            
                cb_handel = plt.colorbar(cb4, ax=ax, label='Norm contri Value')
                cb_handel.set_label('Norm contri Value', size=18)  # Correct way to set the label

                #rows = contri_normalized.shape[0] #epochs
                # for i in range(rows):
                #     min_index = np.argmin(contri_normalized[i, :])
                #     # 这里画的位置需要反一下
                #     rect = plt.Rectangle(( min_index-0.5,i-0.5), 1, 1, edgecolor='k', facecolor='none',linestyle='--', linewidth=2)
                #     ax.add_patch(rect)
                record_inter =kwagrs['record_interve']
                ax.set_xlabel(f'Epoch x{record_inter}',fontsize=15)
                ax.set_ylabel('Subnets',fontsize=15)
                ax.set_title('Normalized contri Values per Epoch for Mscale',fontsize=18)
                ax.tick_params(axis='x', labelsize=14)
                ax.tick_params(axis='y', labelsize=14)

               
            if i ==5: #开始画omega 演化

                # 第一列是epoch
                cb5 = ax.matshow(omega_value[:,1:],cmap='rainbow', aspect='auto', vmin=0, vmax=np.max(omega_value[:,1:]))
                # 设置轴标签和标题
    
                # 添加颜色条
                
                cb5_handle = plt.colorbar(cb5, ax=ax)
                cb5_handle.set_label('Omega Value', size=18)
                ax.set_ylabel(f'Epoch x{record_inter}',fontsize=15)
                ax.set_xlabel('Subnets omegas',fontsize=14)
                ax.set_title('Omegas of Mscale',fontsize=18)
  
               

    
                
                                 
               
                

        return self.fig, self.axes,[cb1,cb2,cb3,cb4,cb5]

    # 使用示例
if __name__ == '__main__':
    Plot_Adaptive1= Plot_Adaptive()
    fig, axes = Plot_Adaptive1.create_subplot_grid2(3, 3)  # 举例：3 行，4 列
    plt.show()
