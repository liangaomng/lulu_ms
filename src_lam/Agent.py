import numpy as np
import torch
from .nn_class import  Multi_scale2,Single_MLP,MoE_Multi_Scale
from torch import optim
from torch import nn
from abc import abstractmethod
from src_lam.xls2_object import Return_expr_dict
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from .excel2yaml import Excel2yaml
from .Analyzer import Analyzer4scale
from .Plot import Plot_Adaptive
from shutil import copy
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from .Analyzer import Analyzer4scale
#DDE_BACKEND=pytorch python Expr3_run.py  
class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        x = torch.tanh(self.linearIn(x)) # Match dimension
        for layer in self.linear:
            x_temp = torch.tanh(layer(x))
            x = x_temp
        
        return self.linearOut(x)
class Expr():
    def __init__(self):
        self.model=None
        self.Read_set_path=None
        self.Save_Path=None
    @abstractmethod
    def _Random(self,seed):
        pass
    @abstractmethod
    def _read_arg_xlsx(self,path):
        pass

    @abstractmethod
    def Prepare_model(self,args):
        pass
    @abstractmethod
    def Valid(self,**kwargs):
        pass
    @abstractmethod
    def Train(self):
        pass
    @abstractmethod
    def Test4Save(self,**kwargs):
        pass
    @abstractmethod
    def Do_Expr(self):
        pass
    @abstractmethod
    def _CheckPoint(self,**kwargs):
        pass
class Base_Args():
    def __init__(self):
        self.model=None
        self.lr=None
        self.seed=None
        self.epoch=None
        self.Train_Dataset=None
        self.Valid_Dataset=None
        self.Test_Dataset=None
        self.Save_Path=None
        self.batch_size=None

        self.Con_record=None
        self.Loss_Record_Path = None
        self.Con_Record_Path=None
        self.Omega_Record_Path=None
        self.Gates_Record_Path=None
        self.w_gates_Record_Path = None
        
        self.data_source=None #task
        self.fig_record_interve=None

        self.Task=None
        
        self.Domain_points=0
        self.Boundary_points=0
        self.Test_points=0
        

    @abstractmethod
    def Layer_set(self,layer_set:list):
        pass

    @abstractmethod
    def Act_set(self,act_set:list):
        pass

    @abstractmethod
    def Ini_Set(self,ini_set:list):
        pass
    @property
    def note(self):
        return self._note
    @note.setter
    def note(self,note):
        self._note=note
class Multi_scale2_Args(Base_Args):
    def __init__(self,scale_coff:list):
        '''
        eg scale_coff: [1,2,3] 子网络的系数
        '''
        super().__init__()
        assert type(scale_coff)==list, "scale coeff must be a list"

        self.subnets_number=len(scale_coff)#  确定后面数组的行，如果4个尺度，就是4行
        self.Scale_Coeff = scale_coff
        self.Act_Set_list = []
        self.Layer_Set_list = []
        self.Ini_Set_list = []
        self.Residual_Set_list = []
        self.Save_Path=None
        self.penalty=None
        self.Boundary_samples=None
        self.All_samples=None

    def Layer_set(self,layer_set:list)->list: #单子网络
        self.Layer_Set_list=layer_set #[1, 10, 10, 10, 1]
    def Act_set(self,act_list)->np.ndarray:
        self.Act_Set_list=act_list #
    def Ini_Set(self,ini_set:list)->np.ndarray:
        #'Ini_Set': ['xavier_uniform']
        self.Ini_Set_list=ini_set
    def __str__(self):
        return  f"Multi_scale2_Args: {self.__dict__}"

class Expr_Agent(Expr):

    def __init__(self,**kwargs):
        '''
            data_source: deepxde/selfpde
            save_folder: name for the folder to save the expr
s
        '''
        super().__init__()

        config = kwargs["config"]
        folder_name = kwargs["save_folder"]
        yaml_path=kwargs["yaml_path"]

        self.args = self._read_arg(config,folder_name,yaml_path = yaml_path)



        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None

        # 需要利用yaml参数
        self.args.epoch = config['SET'][0]['Epoch']
        
        self._Random(seed=self.args.seed)
        self._Check()

        if self.args.Task == "deepxde":
            
            print("deepxde_model") #不用准备dataloader
            self.model = self.Prepare_model()
            
        else:
            self.model = self.Prepare_model()
            

        self.plot = Plot_Adaptive() # 画图
        self.solver=kwargs["solver"]
        
        data=self.solver.Get_Data(domain_numbers=self.args.Domain_points,
                             boundary_numbers=self.args.Boundary_points,
                             test_numbers=self.args.Test_points)

    def _read_arg(self,config:dict,floder_name:str,**kwargs)->Multi_scale2_Args:

        '''
        Here we handle with the args with the yaml file
        '''
        #共同参数
        print("args",config["SET"][0]['Scale_Coeff'])
        scale_list = [float(x) for x in config["SET"][0]['Scale_Coeff'].split(',')]
        section = [int(x) for x in config["SET"][0]['Section'].split(',')]
        act_list = [str(x) for x in config["Subnet"][0]['Act_Set'].split(',')]
        layer_list = [int(x) for x in config["Subnet"][0]['Layer_Set'].split(',')]
        Init_list =[str(x) for x in config["Subnet"][0]['Ini_Set'].split(',')]
        Residual_list =config["Subnet"][0]['Residual']

        args = Multi_scale2_Args(scale_coff=scale_list)
    
        args.model = config["SET"][0]['Model']
        args.lr = config["SET"][0]['lr']
        args.seed=int(config["SET"][0]['Seed'])
        args.epoch=int(config["SET"][0]['Epoch'])
        args.fig_record_interve = int(config["SET"][0]['Fig_Record_Interve'])
        args.record_interve = int(config["SET"][0]['Record_Interve'])
        args.Save_Path= config["SET"][0]['Save_Path'] + floder_name

        print("save path:",args.Save_Path)
        args.batch_size=int(config["SET"][0]['Batch_size'])

        args.section = section #断面保存

        args.Loss_Record_Path = args.Save_Path + "/loss.npy"
        args.Con_Record_Path = args.Save_Path + "/contribution.npy"
        args.Omega_Record_Path = args.Save_Path + "/omegas.npy"
        args.Gates_Record_Path = args.Save_Path + "/gates.npz"
        args.w_gates_Record_Path = args.Save_Path + "/w_gates.npy"

        args.Learn_scale = config["SET"][0]['Learn_scale']
        args.PDE_py = config["SET"][0]['PDE_Solver']
        args.Agent_py = config["SET"][0]['Src_agent']
        args.yaml_file = kwargs["yaml_path"]

        args.MOE = config["SET"][0]["MOE"]
        args.Task = config["SET"][0]["Task"]

        if args.MOE:  
            
            args.sp_experts = int (config["SET"][0]["sp_k"])
            print(f"using moe and sparse experts:",args.sp_experts)

        
        #不一样的任务
        if args.Task == "fitting":#拟合任务
            # path about .pt
            args.Train_Dataset= config["SET"][0]['Train_Dataset']
            args.Valid_Dataset= config["SET"][0]['Valid_Dataset']
            args.Test_Dataset= config["SET"][0]['Test_Dataset']

            self._valid_dataset = torch.load(self.args.Valid_Dataset)
            self._test_dataset = torch.load(self.args.Test_Dataset)
            self._train_dataset = torch.load(self.args.Train_Dataset)

            self._train_loader = DataLoader(dataset=self._train_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=True)
            self._valid_loader = DataLoader(dataset=self._valid_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=True)
            self._test_loader = DataLoader(dataset=self._test_dataset,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)
            args.penalty_data = config["SET"][0]['Penalty_data']

        #pde 任务
        elif args.Task =="deepxde" or "selfpde":
            args.penalty_boun = config["SET"][0]['Penalty_boundary']
            args.penalty_pde = config["SET"][0]['Penalty_pde']
            args.penalty_data = config["SET"][0]['Penalty_data'] #0
           
            args.Boundary_points = config["SET"][0]['Boundary_points']
            args.Domain_points = config["SET"][0]['Domain_points']
            args.Test_points = config["SET"][0]['Test_points']

        # 收集子网络的信息,这里我们假设结构都一样
        for i in range(int(args.subnets_number)):

            args.Act_Set_list.append(act_list)
            args.Layer_Set_list.append(layer_list)
            args.Ini_Set_list.append(Init_list)
            args.Residual_Set_list.append(Residual_list)


        return args
    
    def _Random(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    def Prepare_model(self):

        scale_omegas_coeff = self.args.Scale_Coeff #[1,2,3]
        layer_set = self.args.Layer_Set_list#[1, 10, 10, 10, 1]
        multi_net_act = self.args.Act_Set_list #[4,3] 4个子网络，每个3层激活

        multi_init_weight = self.args.Ini_Set_list #[4,3] 4个子网络，每个3层初始化
        sub_layer_number = len(scale_omegas_coeff)
        residual_en = self.args.Residual_Set_list #[4,3] 4个子网络，每个3层残差
    

        if self.args.model == "mscalenn2":
        
            
            # 普通的多尺度
            self.model = Multi_scale2(
                                        sub_layer_number = np.array(sub_layer_number),
                                        layer_set = np.array(layer_set[0]),#实际是4个list，每个list是一个子网络的神经元
                                        act_set = np.array(multi_net_act),
                                        ini_set = np.array(multi_init_weight),
                                        residual= residual_en[0],
                                        scale_number=scale_omegas_coeff,
                                        scale_learn=self.args.Learn_scale,#scale 学习
                                        )
                
    
        if self.args.model == "MOE":
            #MOE
                self.model = MoE_Multi_Scale( sub_layer_number = np.array(sub_layer_number),
                    layer_set = np.array(layer_set[0]),#实际是4个list，每个list是一个子网络的神经元
                    act_set = np.array(multi_net_act),
                    ini_set = np.array(multi_init_weight),
                    residual= residual_en[0],
                    scale_number=scale_omegas_coeff,
                    scale_learn=self.args.Learn_scale,
                    sparse_experts= self.args.sp_experts)#scale 学习)

        return self.model

    def _Check(self):
        # 检查读取路径
        if self.args.Task == "fitting":
            if not os.path.exists(self.args.Train_Dataset):
                raise FileNotFoundError("Train_Dataset not found")

        # 检查保存路径, 如果没有就创建一个
        if not os.path.exists(self.args.Save_Path):
            os.makedirs(self.args.Save_Path)
        

    def _save_gates_record(self,**kwargs):
        
        pde_gates = kwargs["p_gates"]
        bc_gates = kwargs["b_gates"]
        sub_omega = kwargs["sub_omega"]
        # 设置画布和子图的布局
        #一个2x2的子图布局
        fig, axs = plt.subplots(2, 2, figsize=(25, 12))

        # 展平axs数组以便轻松索引
        axs = axs.flatten()

        #colobar
        im=axs[0].imshow(pde_gates, cmap='bwr',aspect="auto",vmin=0,vmax=0.5)
        axs[0].set_title("pde_gates",fontsize=18 )
        axs[1].imshow(bc_gates, cmap='bwr',aspect="auto",vmin=0,vmax=0.5)
        axs[1].set_title("bc_gates",fontsize=18)
        axs[2].set_xlabel('subnets',fontsize=15)
        # 在特定的列上绘制虚线
        num_columns = pde_gates.shape[1]
        for col in range(num_columns):
            axs[0].axvline(x=col, color='k', linestyle='--', linewidth=1)
            axs[1].axvline(x=col, color='k', linestyle='--', linewidth=1)
        # 使用ax[2]来画条形图
        x_positions = np.arange(len(sub_omega))  # x位置为sub_omega中元素的索引
        axs[2].bar(x=x_positions, height=sub_omega, label="omega value")  # 画条形图
        axs[2].set_title("Scale Value and load",fontsize=18)
        axs[2].set_xlabel('subnets',fontsize=15)
        axs[2].set_ylabel('Value')
        #负载
        load = self.model.Moe_scale._record_load()
        # 创建一个共享x轴但是有第二个y轴的子图
        ax_right = axs[2].twinx()
        # 计算每个点的y值（即每个数组的长度）
        y_values = [arr.shape[0] for arr in load]

        # 注意：这里我们使用了matplotlib的颜色映射。更亮的颜色表示更大的y值。
        colors = plt.cm.rainbow(np.array(y_values) / max(y_values))
        # 绘制散点图，其中颜色亮度基于y值
        for i, y in enumerate(y_values):
            if i == 0:
                ax_right.scatter(x=i, y=y, color=colors[i],alpha=0.8,s=100,label="load")
            else:
                ax_right.scatter(x=i, y=y, color=colors[i],s=100,alpha=0.8)
        # 添加虚线
        # 使用plot函数绘制线条，并设置线型为'--'表示虚线
        ax_right.plot(range(len(y_values)), y_values, '--', color='k')
        axs[2].legend(loc="upper left",fontsize=14)
        ax_right.legend(loc="upper right",fontsize=14)
        # 添加颜色条
        cbar = fig.colorbar(im, ax=axs[1])  # 将颜色条关联到显示图像的轴上
        cbar.set_label('Gates Distribution',fontsize=15)  # 设置颜色条的标签
        
        
         # 定义归一化对象，它会把y_values的值映射到0-1的范围内
        norm = Normalize(vmin=np.min(sub_omega), vmax=np.max(sub_omega))
        color_subnet = plt.cm.rainbow(norm(sub_omega))
        

        for i, y in enumerate(y_values):
            coord_x=load[i][:,0].cpu().detach().numpy()

            coord_y=load[i][:,1].cpu().detach().numpy()
            axs[3].scatter(coord_x,coord_y, color=color_subnet[i],alpha=0.4,s=40,label=f"subnet_{i}")
        # 调整图例
        lgd = axs[3].legend(loc='upper left', bbox_to_anchor=(1.14, 1),fontsize=14) # 将图例移出图表外部

        # 创建颜色条
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        sm.set_array([])
        # 创建颜色条
        cbar = fig.colorbar(sm, ax=axs[3], orientation='vertical')
        # 设置颜色条的标签和字体大小
        cbar.set_label('Scale Value', fontsize=15)


         # 调整整个画布布局，为颜色条留出空间
        plt.subplots_adjust(right=0.9)
        plt.tight_layout()
        
        plt.savefig('{}/gates_{}.png'.format(self.args.Save_Path, kwargs["epoch"]),dpi=300)
        plt.close()

    def _update_loss_record(self, epoch,train_loss,moe_load,**kwargs):

        
    
        
        # 保存画loss的值
        if epoch % self.args.record_interve == 0:
            if self.args.Task == "fitting": 
                valid_loss=self._Valid(epoch=epoch,num_epochs=self.args.epoch)
                test_loss =self._Test4Save(epoch=epoch)
                record = np.array([[epoch, train_loss, valid_loss, test_loss]])
            #deepxde 在外面算test loss包括pde loss/data loss/ bc loss
            elif self.args.Task == "deepxde":
                pde_loss=kwargs["pde_loss"]
                bc_loss=kwargs["bc_loss"]
                data_loss=kwargs["data_loss"]
                moe_loss=kwargs["moe_loss"]
                train_loss=train_loss.detach().cpu().numpy()
                test_loss=kwargs["test_loss"]
                test_data=kwargs["test_data"]
                if self.args.MOE == True:
                    # 记录
                    self.args.sub_omega= [self.model.Moe_model.experts[i].scale_coeff.item() for i in range(len(self.args.Scale_Coeff))]
                else:
                     self.args.sub_omega= [self.model.Multi_scale[i].scale_coeff.item() for i in range(len(self.args.Scale_Coeff))]
                # 假设所有 scale_coeff 都需要被提取
                self.args.sub_omega = np.array( self.args.sub_omega)

            
                record = np.array([[epoch, train_loss,test_loss,pde_loss,bc_loss,data_loss,moe_loss]])

            
            # 检查文件是否存在
            if not os.path.isfile(self.args.Loss_Record_Path):
                # 如果文件不存在，初始化一个空数组并保存
                np.save(self.args.Loss_Record_Path, record)
            else:
                # 读取现有文件
                existing_data = np.load(self.args.Loss_Record_Path)
                # 过滤掉与当前 epoch 相同的记录
                existing_data = existing_data[existing_data[:, 0] != epoch]
                # 将新记录追加到现有数据中
                updated_data = np.vstack((existing_data, record))
                # 保存更新后的数据
                np.save(self.args.Loss_Record_Path, updated_data)

            self._CheckPoint(epoch=epoch)
        #计算contri的值 moe记录门控
        if epoch % 10 == 0:
            if self.args.MOE == False : # 普通的记录contri
                analyzer = Analyzer4scale(model=self.model,d=2,scale_coeffs=self.args.Scale_Coeff)
                contributions = analyzer._analyze_scales()


                # 先创建一个包含 epoch 和 contributions 所有值的列表
                combined_data = [epoch] + contributions     
                # 然后将这个列表转换成一个 numpy 数组，并确保它是二维的
                record = np.array([combined_data])

                if not os.path.isfile(self.args.Con_Record_Path):

                    # Save both arrays into a single npz file.
                    np.save(self.args.Con_Record_Path, record)


                else:

                    existing_data = np.load(self.args.Con_Record_Path)
                    existing_data = existing_data[existing_data[:, 0] != epoch]
                    updated_data = np.vstack((existing_data, record)
                    )
                    np.save(self.args.Con_Record_Path, updated_data)
                    
            if self.args.MOE == True : # MOE 记录gates
                p_gates = kwargs["p_gates"] #[6400,9]
                print("p",p_gates.shape)
                b_gates = kwargs["b_gates"] #[2500,9]


            
                if not os.path.isfile(self.args.Gates_Record_Path):

                    np.savez(self.args.Gates_Record_Path, epoch = epoch, p_gates = p_gates, b_gates = b_gates)

                else:

                    with np.load(self.args.Gates_Record_Path, allow_pickle=True) as existing_data:
                        # Extract existing records
                        existing_epochs = existing_data['epoch']
                        existing_p_gates = existing_data['p_gates']
                        existing_b_gates = existing_data['b_gates']
                        print("exist_p",existing_p_gates.shape)
                        

                        
                        # Filter out the records for the current epoch, if they exist
                        filter_idx = existing_epochs != epoch
                        updated_epochs = existing_epochs[filter_idx]
                        updated_p_gates = existing_p_gates[filter_idx]
                        updated_b_gates = existing_b_gates[filter_idx]
                       
                        # Append the new record for the current epoch
                        updated_epochs = np.append(updated_epochs, epoch)
                        updated_p_gates = np.append(updated_p_gates, [p_gates], axis=0)
                        updated_b_gates = np.append(updated_b_gates, [b_gates], axis=0)
                 
                        
                        # Save the updated records back to the .npz file
                        np.savez(self.args.Gates_Record_Path, epoch=updated_epochs, p_gates=updated_p_gates, b_gates=updated_b_gates)

        #记录omega
        if epoch % 10 == 0:
            
            combine_data = [epoch] + self.args.sub_omega.tolist()

            record = np.array([combine_data])

            if not os.path.isfile(self.args.Omega_Record_Path):

                np.save(self.args.Omega_Record_Path, record)

            else:
                existing_data = np.load(self.args.Omega_Record_Path)
                existing_data = existing_data[existing_data[:, 0] != epoch]
                updated_data = np.vstack((existing_data, record)
                )
                np.save(self.args.Omega_Record_Path, updated_data)
                
        #画图的间隔久
        if epoch % self.args.fig_record_interve == 0:

            self._save4plot(epoch, test_loss,test_data=test_data,
                            type="deepxde",sub_omega=self.args.sub_omega,
                            load=moe_load)
                
            
                

    def _Valid(self,**kwargs):

        epoch = kwargs["epoch"]
        num_epochs = self.args.epoch
        self.model.eval()
        criterion = nn.MSELoss()

        with torch.no_grad():  # 在验证过程中不计算梯度
            sum_val_loss = 0.0
            for inputs, labels in  self._valid_loader:
                inputs= inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                val_loss = criterion(outputs, labels)
                sum_val_loss += val_loss.item()
        avg_val_loss = sum_val_loss / len( self._valid_loader)

        print(f'Epoch [{epoch }/{num_epochs}] Val Loss: {avg_val_loss:.4f}')
        return avg_val_loss
    
    def Train(self):

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        self.model=self.model.to(self.device)

        print(f"we are using device {self.device}")
        for epoch in range(0,self.args.epoch,1):

            epoch_loss = 0.0
            for i, (x, y) in enumerate( self._train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            aver_loss = epoch_loss / len( self._train_loader)

            print('epoch: {}, train loss: {:.6f}'.format(epoch, aver_loss))

            self._update_loss_record(epoch, train_loss=aver_loss)
    
    def _Test4Save(self,**kwargs):

        epoch = kwargs["epoch"]
        self.model.eval()
        criterion = nn.MSELoss()

        with torch.no_grad():
            sum_test_loss = 0.0

            for inputs, labels in self._test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                test_loss = criterion(outputs, labels)
                sum_test_loss += test_loss.item()

        avg_test_loss = sum_test_loss / len(self._test_loader)

        print(f'Test Loss: {avg_test_loss:.6f}')
        return avg_test_loss
    
    def _save4plot(self,epoch,avg_test_loss,**kwargs):
        # 创建两个子图，上下布局
        avg_test_loss= avg_test_loss
        type= kwargs["type"]

        if type != "deepxde":
            # 加载测试数据集
            test_data = torch.load(self.args.Test_Dataset)


        if type == "deepxde":
            test_data = kwargs["test_data"] 
            # 将TensorDataset转换为numpy数组
            x_test = test_data.cpu().numpy() #[..,2]

        # 读取损失记录
        loss_record_npy = np.load(self.args.Loss_Record_Path, allow_pickle=True)
        #读取贡献值
        if self.args.MOE == False:
            contr_record_npy= np.load(self.args.Con_Record_Path, allow_pickle=True)
        #读取sub_omega
        sub_omage_npy = np.load(self.args.Omega_Record_Path, allow_pickle=True)
        #读取gates-test
        if self.args.MOE == True :
            gates_npz = np.load(self.args.Gates_Record_Path, allow_pickle=True)
            # 假设data是一个字典，提取p_gates和bc_gates
            p_gates = gates_npz['p_gates']# 使用 .item() 将numpy对象转换为字典，如果它是字典的话
            b_gates = gates_npz['b_gates']
            if epoch > 0:
                p_gates=p_gates[-1,:,:].reshape(-1,len(self.args.Scale_Coeff))
                b_gates=b_gates[-1,:,:].reshape(-1,len(self.args.Scale_Coeff))
            if epoch == 0:
                
                p_gates=p_gates.reshape(-1,len(self.args.Scale_Coeff)) #(6400,0)
                b_gates=b_gates[:,:].reshape(-1,len(self.args.Scale_Coeff))

        if x_test.shape[-1] == 2: #[5000,2]
            # analyzer
           
            if self.args.MOE == False:
                fig, axes,cb_list = self.plot.plot_2d(nrow=4,ncol=3,
                                            loss_record=loss_record_npy,
                                            contr_record=contr_record_npy,
                                            omega_record=sub_omage_npy,
                                            epoch=epoch,
                                            avg_test_loss=avg_test_loss,
                                            solver=self.solver,
                                            model=self.model,
                                            record_interve =self.args.record_interve,
                                            MOE=self.args.MOE,
                                            )
            if self.args.MOE == True:
            
                fig,axes,cb_list = self.plot.plot_moe__loss_gates(nrow=3,ncol=3,
                                                            loss_record = loss_record_npy,
                                                            omega_record=sub_omage_npy,
                                                            epoch=epoch,
                                                            model=self.model,
                                                            solver=self.solver,
                                                            record_interve = self.args.record_interve,
                                                            p_gates= p_gates,
                                                            b_gates= b_gates,
                                                            load = kwargs["load"]
                                                            )
                # 这里我们默认专家网络的平方关系，比如9、25
                nrow = int(np.sqrt (self.args.subnets_number))

                fig_load,axes_load = self.plot.plot_moe__load(nrow=nrow,
                                                              ncol=nrow,epoch=epoch,
                                                              load=  kwargs["load"],
                                                              name = self.solver.name,
                                                             )

                fig_load.savefig('{}/combined_load_{}.png'.format(self.args.Save_Path, epoch),bbox_inches='tight')
                       # 关闭当前的图形窗口
                for ax in axes_load:
                    ax.cla()     
                plt.close()
                
                
                
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14

         
          
        # 保存整个图表
        fig.savefig('{}/combined_loss_{}.png'.format(self.args.Save_Path, epoch),
                    bbox_inches='tight')
        
        self.plot.fig=None
        self.plot.fig_load = None
        
        for cb in cb_list:
            cb.remove()
        # 关闭当前的图形窗口
        for ax in axes:
            ax.cla()
        plt.close()
        


    def _CheckPoint(self,**kwargs):
            epoch=kwargs["epoch"]
            dir_name=self.args.Save_Path+"/"+self.args.model+".pth"
            if os.path.exists(self.args.Save_Path):
                pass
            else:
                os.mkdir(self.args.Save_Path)

            torch.save(self.model.state_dict(), dir_name)
            print(f"save model at epoch {epoch}")

    def Train_PDE(self):

        self.model = self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.lr)
        criterion = nn.MSELoss()
        boundary_loss=nn.MSELoss()
        start_b_index= self.args.All_samples-self.args.Boundary_samples #bondary index

        for epoch in range(self.args.epoch):
            #every epoch to sample in PDE_solver
            sample=self.solver.sample(self.args.batch_size)
            sample=torch.from_numpy(sample).float().to(self.device)
            inputs = sample[:,:,0:2]
            labels = sample[:,:,2:3]
            #predict
            outputs = self.model(inputs)#[batch,5000,1]
            boundary_pred=outputs[:,start_b_index:,0]
            boundary_label =  labels[:,start_b_index:,0].to(self.device)
            #assert  #[batch,2600]
            assert boundary_pred.shape[-1] == self.args.Boundary_samples
            #boundary loss
            b_loss = self.args.penalty_boun * boundary_loss(boundary_pred, boundary_label)
            #domian loss
            loss1=criterion(labels[:,:start_b_index,0], outputs[:,:start_b_index,0])
            loss = loss1+b_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()

            aver_loss=epoch_loss/self.args.batch_size
            print("epoch:{},aver_loss:{:.6f}".format(epoch,aver_loss),flush=True)

            self._update_loss_record(epoch, train_loss=aver_loss)
   
   
    def Ritz(self):
        
        # Network structure
        params = dict()
        params["d"] = 2 # 2D
        params["dd"] = 1 # Scalar field
        params["bodyBatch"] = 1024 # Batch size
        params["bdryBatch"] = 1024 # Batch size for the boundary integral
        params["lr"] = 0.01 # Learning rate
        params["preLr"] = 0.01 # Learning rate (Pre-training)
        params["width"] = 8 # Width of layers
        params["depth"] = 2 # Depth of the network: depth+2
        params["numQuad"] = 40000 # Number of quadrature points for testing
        params["trainStep"] = 50000
        params["penalty"] = 100
        params["preStep"] = 0
        params["diff"] = 0.001
        params["writeStep"] = 200
        params["sampleStep"] = 10
        params["step_size"] = 500
        params["gamma"] = 0.3
        params["decay"] = 0.00001
        print("ritz")
        model = RitzNet(params)
        return model
    def _update_weights(self,epoch,w_gates):
        
        # 检查文件是否存在
        if epoch % self.args.record_interve == 0:
          
            if not os.path.isfile(self.args.w_gates_Record_Path):
                # 如果文件不存在，初始化一个空数组并保存
                np.save(self.args.w_gates_Record_Path, w_gates)
            else:
                # 读取现有文件
                existing_data = np.load(self.args.w_gates_Record_Path)
                # 过滤掉与当前 epoch 相同的记录
                existing_data = existing_data[existing_data[:, 0] != epoch]
                # 将新记录追加到现有数据中
                updated_data = np.vstack((existing_data, w_gates))
                # 保存更新后的数据
                np.save(self.args.w_gates_Record_Path, updated_data)
            
        if epoch % self.args.fig_record_interve == 0:
            
            plt.matshow(w_gates,cmap="jet",vmin=-1,vmax=1)

            plt.colorbar()
            plt.title(f"epoch={epoch} experts gates' weights",fontsize=18)
            plt.savefig(f"{self.args.Save_Path}/weights_{epoch}.png",dpi=200)
            plt.close()
        
        
        
   
    def Train_XDE(self):

        #Copy the sover and agent to save
        des_file = self.args.Save_Path
        soure_file_solver = self.args.PDE_py
        soure_file_agent = self.args.Agent_py
        source_file_yaml = self.args.yaml_file
        copy(soure_file_solver,des_file)
        copy(soure_file_agent,des_file)
        copy(source_file_yaml,des_file)
        print("save solevr, agent and yaml")


        self.model = self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(),self.args.lr)
        
        print("train_data_penalty:{}".format(self.args.penalty_data))
        print("train_pde_penalty:{}".format(self.args.penalty_pde))
        print("train_bc_penalty:{}".format(self.args.penalty_boun))

                    
        for epoch in range(self.args.epoch):
          
            self.solver.data.train_x,_,_=self.solver.data.train_next_batch()
            self.solver.data.train_x_all=self.solver.data.train_points()
            


            # deepxde重新生成边界条件点-every epoch
            all_data= self.solver.data.train_x_all #pde+bc data
     

            bc_data= self.solver.data.train_x_bc #bc data
   
            # 将 bc_data 转换为一组元组以进行比较
            bc_set = set(map(tuple, bc_data))

            # 保留不在 bc_set 中的 pde_data 行
            pde_data = np.array([row for row in all_data if tuple(row) not in bc_set])

            train_data= self.solver.data.train_x#pde+bc

            train_data=torch.from_numpy(train_data).float().to(self.device)
            #inputs
            pde_data=torch.from_numpy(pde_data).float().to(self.device)
            pde_data.requires_grad=True
            bc_data = torch.from_numpy(bc_data).float().to(self.device)
            bc_data.requires_grad=True
            #pde loss
            if self.args.penalty_pde != 0:
                train_pde_loss,pde_moe ,p_gates,_,_= self.solver.pde_loss(net=self.model,pde_data=pde_data,MOE=self.args.MOE)
    
                #bc loss
                train_bc_loss,bc_moe,b_gates,_,_ = self.solver.bc_loss(net=self.model,data=bc_data,MOE = self.args.MOE)
                train_loss =  self.args.penalty_pde*train_pde_loss + pde_moe + bc_moe
                train_loss += self.args.penalty_boun*train_bc_loss 
            #data loss
            elif self.args.penalty_pde == 0: #pure data_driven
                train_data_loss,data_moe,_,_,_=self.solver.data_loss(net=self.model,data=train_data,MOE=self.args.MOE)

                print("train_data_loss:{}".format(train_data_loss)) 
                train_loss = train_data_loss *self.args.penalty_data + data_moe
            
            #train
            optimizer.zero_grad()
            train_loss.backward()
      
            #clip -2范数-长度
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()



            #plot -test 主要看data，pde和bc只是辅助
            with torch.no_grad():
                
                if epoch % 1== 0:
                    test=self.solver.data.test_x
                    torch_test=torch.from_numpy(test).float().to(self.device)

                    data_loss,moe_loss,d_gates,d_w_gates,d_load = self.solver.data_loss(net=self.model, 
                                                                                        data=pde_data,MOE=self.args.MOE)

                    #test loss only see the data loss
                    test_loss= data_loss 
            
                    
                    print(f"epoch={epoch} pde_loss:{train_pde_loss:.2e},bc{train_bc_loss:.2e},data_loss:{data_loss:.2e},moe_loss:{moe_loss:.2e}",flush=True)
            
            
                if self.args.MOE:

                    #data 的moe
                    
                    Moe_load = d_load
                    self._update_weights(epoch, w_gates = d_w_gates.cpu().detach().numpy())
                   

                    # d_w_gates d_gates的保存

                    self._update_loss_record(   epoch, train_loss = train_loss,
                                                pde_loss = train_pde_loss.item(),
                                                bc_loss = train_bc_loss.item(),
                                                data_loss = data_loss.item(),
                                                test_loss = test_loss.item(),
                                                test_data = torch_test,
                                                moe_load = Moe_load,
                                                moe_loss=moe_loss.item(),
                                                p_gates = p_gates.cpu().detach().numpy(),
                                                b_gates = b_gates.cpu().detach().numpy(),
                                               ) 
                else:
                    self._update_loss_record(   epoch, train_loss=train_loss,
                                                pde_loss=train_pde_loss.item(),
                                                bc_loss=train_bc_loss.item(),
                                                data_loss=data_loss.item(),
                                                test_loss=test_loss.item(), 
                                                test_data=torch_test,
                                                moe_load =None,

                                                moe_loss= torch.tensor(0),
                                                p_gates = torch.tensor(0),
                                                b_gates = torch.tensor(0),
                                                )
            

                
                
    def Do_Expr(self):
        
        print("self.args.Task",self.args.Task)
        if self.args.Task == "selfpde":
            print("this is the selfpde task")
            self.Train_PDE()
        elif self.args.Task == "deepxde":
            print("this is the deepxde task")
            self.Train_XDE()
        elif self.args.Task == "fitting":#fitting
            print("this is the fitting task")
            self.Train()

        print("we have done the expr")

if __name__=="__main__":
    pass