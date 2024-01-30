import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from .Analyzer import Analyzer4scale
import numpy as np

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
        for r in range(nrow - 1):
            ax = self.fig.add_subplot(gs[r, :])
            self.axes.append(ax)

        # 添加最后一行的子图
        for c in range(ncol):
            ax = self.fig.add_subplot(gs[-1, c])
            
         
            self.axes.append(ax)
          

    def _create_subplot_grid2(self,nrow, ncol):
        self.fig = plt.figure(figsize=(1.6 * ncol * 3, 1.4 * nrow * 3))
        gs = GridSpec(nrow, ncol, figure=self.fig, hspace=0.4, wspace=0.3)
        self.axes = []

        for c in range(ncol):
            ax = self.fig.add_subplot(gs[0, c])
            self.axes.append(ax)

        # 添加第2行的图
        for r in [1]:
            ax = self.fig.add_subplot(gs[r, :])
            self.axes.append(ax)
        

        # 添加倒数第二行的子图
        for c in range(ncol):
            ax = self.fig.add_subplot(gs[-2, c])
            self.axes.append(ax)
        
        for r in [3]:
            ax = self.fig.add_subplot(gs[r, :])
            ax2=ax.twinx()
            self.axes.append(ax)
            self.axes.append(ax2)

            
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
        analyzer = kwagrs["analyzer"]
        #画图计算的solver
        solver=kwagrs["solver"]
        model=kwagrs["model"]
        if self.fig is None:
            self._create_subplot_grid2(nrow, ncol)
        Record = kwagrs["contribution_record"]
      

        for i, ax in enumerate(self.axes):
        
            if i == 2:  # 第一张图
                # 在第一个子图上绘制预测值的散点图
   
                avg_test_loss = kwagrs["avg_test_loss"]
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
                                                data=solver.data.test_x)
                U_true=U_true.reshape(-1,1)
                
                data=solver.data.test_x
                
                #x轴实际是t【:,1】
                sc=self.axes[2].scatter(data[:,1],data[:,0],c=np.abs(U_pred-U_true),cmap="bwr",s=1.5)
                # 添加颜色条
                cb3=plt.colorbar(sc, ax=self.axes[2], label='Absolute Difference')

                # 隐藏 x 和 y 轴的刻度和标签
                self.axes[2].set_xticks([])
                self.axes[2].set_yticks([])
                # 设置第一个子图的图例、坐标轴标签和标题
                self.axes[2].set_title("Loss={:.6f}_Epoch{}".format(avg_test_loss, epoch))


            if i == 3:  # 第二张图
                # 在第最后子图上绘制损失曲线
            
                loss_record_df = kwagrs["loss_record"]

                ax.plot(loss_record_df[:,0], loss_record_df[:,1], label="Train Loss", color="blue")
                #ax.plot(loss_record_df[:,0], loss_record_df[:,2], label="Valid Loss", color="red")
                ax.plot(loss_record_df[:,0], loss_record_df[:,2], label="Test Loss", color="green",alpha=0.8,marker=">")
                
                ax.plot(loss_record_df[:,0], loss_record_df[:,3], label="PDE Loss", color="purple",alpha=0.8,marker="*")
                ax.plot(loss_record_df[:,0], loss_record_df[:,4], label="Boundary Loss", color="black",marker="+")
                ax.plot(loss_record_df[:,0], loss_record_df[:,5], label="Data Loss", color="#FF4500",marker="o")
            
                # # 画三条虚线
                if epoch >= 1000:
                    for j, value in enumerate(Record):
                        ax.axvline(x=value, color=c_map[j], linestyle='--')
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
                ax.legend(loc="best", fontsize=16)
                # 设置第二个子图的图例、坐标轴标签和标题
                ax.set_yscale('log')  # 将y轴设置为对数尺度
                ax.set_xlabel('Epoch', fontsize=16)
                ax.set_ylabel('Loss', fontsize=16)
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                ax.tick_params(labelsize=16, width=2, colors='black')
                ax.set_title("Loss_Epoch{}".format(epoch))
            if i == 4:  # 第三行图开始画贡献度
                #我们这里记录一些贡献度 随着epoch记录 --山脊
                contri_path=kwagrs["contribution_record_path"]
                omege_value=kwagrs["omega"]
                omega_path=kwagrs["omega_value_path"]
                contributions=analyzer._analyze_scales()
                #Now, to combine them into a single list:
                combined_data = [epoch] + contributions
        
                omege_value_new=[epoch]+omege_value.tolist()
        
                omege_value_array=  np.array([omege_value_new])

        
                # To create a numpy array from this combined data:
                contri_array = np.array([combined_data])  # Note the brackets to create a two-dimensional array

                # 检查文件是否存在
                if not os.path.isfile(omega_path and contri_path):
                    np.save(omega_path,omege_value_array)
                    np.save(contri_path, contri_array)
                    
                else:
                    # 读取现有文件
                    for path,data in zip([contri_path,omega_path],[contri_array,omege_value_array]):
                        
                        existing_data = np.load(path,allow_pickle=True)
                        
                        # 过滤掉与当前 epoch 相同的记录
                        existing_data = existing_data[existing_data[:, 0] != epoch]
                        # 将新记录追加到现有数据中
                        updated_data = np.vstack((existing_data, data))
                        # 保存更新后的数据
                        np.save(path, updated_data)
                if (epoch == Record[0]):
                    analyzer.plot_contributions(ax=self.axes[4], fig=self.fig, cmap=c_map[0])
                
            if i == 5:
                if (epoch == Record[1]):
                    analyzer.plot_contributions(ax=self.axes[5], fig=self.fig, cmap=c_map[1])
            if i == 6:
                if (epoch == Record[2]):
                    # for j in epoch_axv:
                    analyzer.plot_contributions(ax=self.axes[6], fig=self.fig, cmap=c_map[2])
            if i== 7:
               
                if(epoch%10==0):
                    #读取贡献度
                    contri_record_df=np.load(contri_path,allow_pickle=True)
                     
        
                    scale_number=contri_record_df.shape[1]-1 #减去epoch
                    omege_value=np.load(omega_path,allow_pickle=True)#[epoch,3]
 
                    # 定义15种颜色
                    # The above code is creating a list called "colors" that contains several color names.
                    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'pink', 'brown', 'grey', 'lime', 'indigo', 'violet']

                    for i in range(scale_number):
                        self.axes[7].plot(contri_record_df[:,0],
                                          contri_record_df[:,i+1],
                                          label=f"scale_{i+1}_contri",
                                          color=colors[i],
                                          linestyle='--',
                                          zorder=2)
                        
                        self.axes[8].plot(omege_value[:,0],#epoch
                                 omege_value[:,i+1], 
                                 label=f'omega__{i+1}_value', 
                                 marker='>'
                                 ,color=colors[i],
                                 zorder=2,linewidth=1)
                             
                  
                    leg1=self.axes[7].legend(loc="upper left", fontsize=8)
                    leg2=self.axes[8].legend(loc="upper right", fontsize=8)
                    leg1.set_zorder(3)
                    leg2.set_zorder(3)
                    # 设置第二个子图的图例、坐标轴标签和标题\
                    self.axes[7].set_xlabel('Epoch', fontsize=16)
                    self.axes[7].set_ylabel('Contribution', fontsize=8)
                    self.axes[7].set_title("Contribution_scale_Epoch")
                    # For y-axis
                     #找到最大和最小的值
                    # 计算这四列（第二列到第五列）的最大值和最小值
                    min_value_four_columns = np.amin(data[:, 1:])
                    max_value_four_columns = np.amax(data[:, 1:])
                    self.axes[8].set_ylim(min_value_four_columns-1,max_value_four_columns+1)

             
                    

        return self.fig, self.axes,[cb1,cb2,cb3]

    # 使用示例
if __name__ == '__main__':
    Plot_Adaptive1= Plot_Adaptive()
    fig, axes = Plot_Adaptive1.create_subplot_grid2(3, 3)  # 举例：3 行，4 列
    plt.show()
