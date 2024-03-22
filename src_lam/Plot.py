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
        self.fig_load = None
        self.axes_load = None
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
            self.axes.append(ax1)
        ax2 = self.fig.add_subplot(gs[2, mid_point:])  # 第二个子图占据右半边
        self.axes.append(ax2)
        ax3 = self.fig.add_subplot(gs[3, mid_point:])  # 第二个子图占据右半边
        self.axes.append(ax3)



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

    def _create_moe_load_grid2(self,nrow,ncol):

        self.fig_load = plt.figure(figsize=(2.2 * ncol * 3, 1.8 * nrow * 3)) 
        gs = GridSpec(nrow, ncol, figure=self.fig, hspace=0.4, wspace=0.3)
        self.axes_load = []
    # 对于每一行和每一列，添加子图
        for r in range(nrow):
            for c in range(ncol):
                ax =  self.fig_load.add_subplot(gs[r, c])
                self.axes_load.append(ax)




    def plot_moe__loss_gates(self, nrow,ncol,**kwagrs):
        
        if self.fig is None:
            self._create_subplot_moe_grid2(nrow=3, ncol=3)

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
        record_inter =kwagrs['record_interve']

        # 注意：这里我们使用了matplotlib的颜色映射。更亮的颜色表示更大的y值。

        for i, ax in enumerate(self.axes):
            
            if i == 2:
                epoch = kwagrs["epoch"]# 每隔记录10个增长1个

                # 在第一个子图上绘制真实值的散点图

                if solver.name =="Burgers":
                    U_true,cb1,T,X=solver.plot_exact(   ax=self.axes[0],
                                                    title="True",
                                                    cmap="bwr",
                                                    data=solver.data.test_x)

                    # 在第一个子图上绘制真实值的散点图
                    U_pred,cb2=solver.plot_pred(    ax=self.axes[1],
                                                    model=model,
                                                    title="Pred",
                                                    cmap="bwr",
                                                    data=solver.data.test_x,MOE = True)
                                            
                    
                    #x轴实际是t【:,1】
                    sc=self.axes[2].scatter(T,X,c= np.abs(U_pred-U_true),cmap="bwr",s=2)
                    # 添加颜色条
                    cb3=plt.colorbar(sc, ax=self.axes[2], label='Absolute Difference')
                else:                     
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
                                                    data=solver.data.test_x)
                    U_true=U_true.reshape(-1,1)
                    
                    data=solver.data.test_x
                    
                    #x轴实际是【:,1】
                    sc=self.axes[2].scatter(data[:,0],data[:,1],c=np.abs(U_pred-U_true),cmap="bwr",s=2)
                    # 添加颜色条
                    cb3=plt.colorbar(sc, ax=self.axes[2], label='Absolute Difference')

                # 隐藏 x 和 y 轴的刻度和标签
                self.axes[2].set_xticks([])
                self.axes[2].set_yticks([])
                # 设置第一个子图的图例、坐标轴标签和标题
                mse = np.mean((U_pred - U_true) ** 2)
                self.axes[2].set_title("MSE={:.6f}_Epoch{}".format(mse, epoch),fontsize=18) 
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
                

            if i ==4:#  pde gates
                pde_gates = kwagrs["p_gates"]
                im=ax.imshow(pde_gates, cmap='bwr',aspect="auto",vmin=0,vmax=0.5)
                ax.set_title("p_gates",fontsize=18 )
            
                ax.set_xlabel('subnets',fontsize=15)
                # 在特定的列上绘制虚线
                num_columns = pde_gates.shape[1]
                for col in range(num_columns):
                    ax.axvline(x=col, color='k', linestyle='--', linewidth=1)
                cb4 = plt.colorbar(im, ax=ax, label='Prob compare with experts')
                # 设置横坐标刻度的字体大小
   
                 

            if i ==5:
                bc_gates = kwagrs["b_gates"]
                im = ax.imshow(bc_gates, cmap='bwr',aspect="auto",vmin=0,vmax=0.5)
                ax.set_title("bc_gates",fontsize=18)
                ax.set_xlabel('subnets',fontsize=15)
                for col in range(num_columns):
                    ax.axvline(x=col, color='k', linestyle='--', linewidth=1)
                cb5 = plt.colorbar(im, ax=ax, label='Prob compare with experts')
            

            if i == 6:
            
                # 绘制散点图，其中颜色亮度基于y值
                for j, y in enumerate(y_values):
                    color = scalar_map.to_rgba(y)
                    if j == 0:
                        ax.scatter(x=j, y=y, color=color,alpha=0.8,s=100,label="load")
                    else:
                        ax.scatter(x=j, y=y, color=color,s=100,alpha=0.8)
                ax.plot(range(len(y_values)), y_values, '--', color='k')
                
                if epoch>=1:
                    ax.set_title(f" Experts_Points Min Data_loss epoch :{min_epoch * record_inter}")
                cb6=plt.colorbar(scalar_map, ax=ax, label='Load Points in test')


        return self.fig, self.axes,[cb1,cb2,cb3,cb4,cb5,cb6]
    
    def plot_moe__load(self, nrow,ncol,**kwagrs):
        
        if self.fig_load is None :
            self._create_moe_load_grid2(nrow=nrow, ncol=ncol) # 3*3 或5*5

        load = kwagrs['load']
        # 计算每个点的y值（即每个数组的长度）
        y_values = [arr.shape[0] for arr in load]
        epoch = kwagrs["epoch"]# 每隔记录10个增长1个
        #正常画9个子网络的load
        ax_order=0
        name = kwagrs["name"] #考虑x和t的名字

    
                
        for j, y in enumerate(y_values):
            
            coord_x=load[j][:,0].cpu().detach().numpy() #x
            coord_y=load[j][:,1].cpu().detach().numpy() #t 

        
            
            if name =="Burgers":
                
                self.axes_load[ax_order].scatter(coord_y,coord_x,alpha=0.8,s=2,label=f"subnet_{j}_load")
            else:
                
                self.axes_load[ax_order].scatter(coord_x,coord_y,alpha=0.8,s=2,label=f"subnet_{j}_load")
                
            self.axes_load[ax_order].legend(loc="upper left")

            self.axes_load[ax_order].set_title(f"Load{y}_epoch{epoch}",fontsize=20)
            ax_order = ax_order+1

        return self.fig_load, self.axes_load

            
 
    def plot_2d(self, nrow, ncol, **kwagrs):
        # 绘制一些示例数据
        c_map = ["Green", "Purple"]
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
                print(solver.name)
                if solver.name =="Burgers":
                    U_true,cb1,T,X=solver.plot_exact(   ax=self.axes[0],
                                                    title="True",
                                                    cmap="bwr",
                                                    data=solver.data.test_x)

                    # 在第一个子图上绘制真实值的散点图
                    U_pred,cb2=solver.plot_pred(    ax=self.axes[1],
                                                    model=model,
                                                    title="Pred",
                                                    cmap="bwr",
                                                    data=solver.data.test_x,
                                                    MOE=False
                                                    
                                                )
                    
                    #x轴实际是t【:,1】
                    sc=self.axes[2].scatter(T,X,c= np.abs(U_pred-U_true),cmap="bwr",s=2)
                    # 添加颜色条
                    cb3=plt.colorbar(sc, ax=self.axes[2], label='Absolute Difference')
                else:                     
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
                    
                    #x轴实际是【:,1】
                    sc=self.axes[2].scatter(data[:,0],data[:,1],c=np.abs(U_pred-U_true),cmap="bwr",s=2)
                    # 添加颜色条
                    cb3=plt.colorbar(sc, ax=self.axes[2], label='Absolute Difference')

                # 隐藏 x 和 y 轴的刻度和标签
                self.axes[2].set_xticks([])
                self.axes[2].set_yticks([])
                # 设置第一个子图的图例、坐标轴标签和标题
                mse = np.mean((U_pred - U_true) ** 2)
                self.axes[2].set_title("MSE={:.2e}_Epoch{}".format(mse, epoch),fontsize=18)


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
                ax.set_xlabel(f'Epoch x{record_inter}',fontsize=20)
                ax.set_ylabel('Subnets',fontsize=20)
                ax.set_title('Normalized contri Values per Epoch for Mscale',fontsize=18)
                ax.tick_params(axis='x', labelsize=18)
                ax.tick_params(axis='y', labelsize=18)

               
            if i ==5: #画1个bar图,有两个颜色
                #开始的epoch
                contri_epoch0=contri_normalized[0,:]
                print("contri_normalized at epoch 0",contri_epoch0.shape) #(1,9)
                epoch_omega = omega_value[-1, 1:]    
                vmin = np.log(epoch_omega.min()) +0.1 # 使用较大底数的对数
                vmax = np.log(epoch_omega.max())
                omega_norm = Normalize(vmin=vmin, vmax=vmax)
                
                c_map = cm.Greens  # Using a built-in green colormap
                # 用于存储散点的x和y值，以便绘制连接线
                x_vals = []
                y_vals = []
                for j, (contrib, coeff) in enumerate(zip(contri_epoch0.flatten(), epoch_omega)):
                    coeff_norm = omega_norm(coeff) # 应用对数规范化
                    ax.scatter(j, contrib, color=c_map(coeff_norm), label=f'Scale {j}: Coeff {coeff:.1f}')
                    x_vals.append(j)
                    y_vals.append(contrib)
                    #ax.text(j, contrib, f'{coeff:.1f}', ha='center', va='bottom', fontsize=7)     
                #设置图表标题和轴标签
                    # 在所有散点之间绘制连接线
                ax.plot(x_vals, y_vals, color='k', linestyle='-', linewidth=1)
                ax.set_title(f'Contribution per Scale at epoch={0}')
                ax.set_xlabel('Scale net')
                ax.set_ylabel('Contribution',fontsize=10)
                ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
            
            if i ==6: #画min dataloss 的epoch
                # 归一化contri值 因为【epoch,9】，对每一个epoch做归一化 对每一行
                if epoch>=1:
                    min__data_epoch = np.argmin(loss_record_df[:, 5])
                    epoch_omega = omega_value[min__data_epoch, 1:]  #第一列是epoch
                    print("min__data_epoch",min__data_epoch)
    
                    vmin = np.log(epoch_omega.min()) + 0.1 # 使用较大底数的对数
                    vmax = np.log(epoch_omega.max())
                    omega_norm = Normalize(vmin=vmin, vmax=vmax)
                    x_vals_end = []
                    y_vals_end = []
                    c_map = cm.Blues  # Using a built-in Blues colormap
                    contri_min_data = contri_normalized[min__data_epoch,:]

                    for k, (contrib, coeff) in enumerate(zip(contri_min_data.flatten(), epoch_omega)):
                        coeff_norm = omega_norm(coeff) # 应用对数规范化
                        x_vals_end.append(k)
                        y_vals_end.append(contrib)
                        ax.scatter(k, contrib, color=c_map(coeff_norm), label=f'Scale {k}: Coeff {coeff:.1f}')
                        #ax.text(k, contrib, f'{coeff:.1f}', ha='center', va='bottom', fontsize=10)

                    ax.plot(x_vals_end, y_vals_end, color='k', linestyle='-', linewidth=1)
                    #设置图表标题和轴标签
                    ax.set_title(f'Contribution per Scale at min data_loss epoch={min__data_epoch * record_inter}')
                    ax.set_xlabel('Scale net')
                    ax.set_ylabel('Contribution',fontsize=10)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
                else:
                    pass
    
               

    
                
                                 
               
                

        return self.fig, self.axes,[cb1,cb2,cb3,cb4]

    # 使用示例
if __name__ == '__main__':
    Plot_Adaptive1= Plot_Adaptive()
    fig, axes = Plot_Adaptive1.create_subplot_grid2(3, 3)  # 举例：3 行，4 列
    plt.show()
