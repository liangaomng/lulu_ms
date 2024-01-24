import numpy as np
import torch
from .nn_class import  Multi_scale2,Single_MLP
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
        self._note=None
        self.Con_record=None
        self.Loss_Record_Path = None
        self.Con_Record_Path=None
        self.Omega_Record_Path=None
        
        self.PDE=None #task
        self.fig_record_interve=None
        
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
    def __init__(self,scale_coff):
        super().__init__()
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
class Expr_Agent(Expr):
    def __init__(self,pde_task=False,**kwargs):
        super().__init__()

        xls2_dict =Return_expr_dict.sheet2dict(kwargs["Read_set_path"])
        self.args = self._read_arg_xlsx(xls2_dict,pde_task=pde_task) # 读取参数

        Excel2yaml(path=self.args.Save_Path,
                   excel=kwargs["Read_set_path"]).excel2yaml()  # convert 2yaml
        
        self.args.PDE = pde_task   #deepxde

        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        # 外面传入的参数 不用excel
        self.epoch = kwargs["train_epoch"]    
        
        self._Random(seed=self.args.seed)
        self._Check()
        if self.args.PDE == "deepxde":
            print("deepxde_model") #不用准备dataloader
            self.model = self.Prepare_model()
        else:
            self.model = self.Prepare_model()

        self.plot = Plot_Adaptive() # 画图
        self.solver=kwargs["solver"]
        
        data=self.solver.Get_Data(domain_numbers=self.args.Domain_points,
                             boundary_numbers=self.args.Boundary_points,
                             test_numbers=self.args.Test_points)

    def _read_arg_xlsx(self,xls2_object:dict,**kwargs)->Multi_scale2_Args:
        #共同参数
        args=Multi_scale2_Args(xls2_object["SET"].Scale_Coeff)
        args.model=xls2_object["SET"].Model[0]
        args.lr=xls2_object["SET"].lr[0]
        args.seed=int(xls2_object["SET"].SEED[0])
        args.epoch=int(xls2_object["SET"].Epoch[0])
        args.fig_record_interve = int(xls2_object["SET"].Fig_Record_Interve[0])
        args.Save_Path=xls2_object["SET"].Save_Path[0]
        args.batch_size=int(xls2_object["SET"].Batch_size[0])
        args.Con_record=xls2_object["SET"].Con_record #list
        args.Loss_Record_Path = args.Save_Path + "/loss.npy"
        args.Con_Record_Path = args.Save_Path + "/contribution.npy"
        args.Omega_Record_Path = args.Save_Path + "/omegas.npy"
        args.Learn_scale=xls2_object["SET"].Learn_scale[0]
        args.PDE_py=xls2_object["SET"].PDE_Solver[0]
        args.Agent_py=xls2_object["SET"].Src_agent[0]

        
        #不一样的
        if kwargs["pde_task"] == False:#拟合任务
            args.Train_Dataset=xls2_object["SET"].Train_Dataset[0]
            args.Valid_Dataset=xls2_object["SET"].Valid_Dataset[0]
            args.Test_Dataset=xls2_object["SET"].Test_Dataset[0]

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

        if kwargs["pde_task"] =="selfpde":
            args.Valid_Dataset = xls2_object["SET"].Valid_Dataset[0]
            args.Test_Dataset = xls2_object["SET"].Test_Dataset[0]
            args.penalty=xls2_object["SET"].Penalty[0]
            args.Boundary_samples=int(xls2_object["SET"].Sum_Samples[0]-
                                      xls2_object["SET"].Domain_Numbers[0])
            args.All_samples=int(xls2_object["SET"].Sum_Samples[0])
            self._valid_dataset = torch.load(self.args.Valid_Dataset)
            self._train_dataset = None
            self._train_loader = None
            self._valid_loader = DataLoader(dataset=self._valid_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=True)
            self._test_loader = DataLoader(dataset=self._test_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=True)

        if kwargs["pde_task"] =="deepxde":
            self._train_dataset = None
            self._train_loader = None
            self._valid_dataset = None
            self._valid_loader = None
            self._test_dataset=None
            self._test_loader = None

        #loss
        if kwargs["pde_task"]=="deepxde" or kwargs["pde_task"]=="selfpde":
            args.penalty_boun=xls2_object["SET"].Penalty_boundary[0]
            args.penalty_pde=xls2_object["SET"].Penalty_pde[0]
            args.penalty_data=xls2_object["SET"].Penalty_data[0]
            args.Boundary_points=int(xls2_object["SET"].Boundary_points[0])
            args.Domain_points=int(xls2_object["SET"].Domain_points[0])
            args.Test_points=int(xls2_object["SET"].Test_points[0])
        #  收集子网络的信息,这里我们假设子网络都一样
        for i in range(int(args.subnets_number)):
            sub_key="Subnet"
            args.Act_Set_list.append(xls2_object[sub_key].Act_Set)
            args.Layer_Set_list.append(xls2_object[sub_key].Layer_Set)
            args.Ini_Set_list.append(xls2_object[sub_key].Ini_Set)
            args.Residual_Set_list.append(xls2_object[sub_key].Residual)

        return args
    
    def _Random(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    def Prepare_model(self):

        if self.args.model == "mscalenn2":
            scale_omegas_coeff = self.args.Scale_Coeff #[1,2,3]
            layer_set=self.args.Layer_Set_list#[1, 10, 10, 10, 1]
            multi_net_act=self.args.Act_Set_list #[4,3] 4个子网络，每个3层激活

            multi_init_weight=self.args.Ini_Set_list #[4,3] 4个子网络，每个3层初始化
            sub_layer_number=len(scale_omegas_coeff)
            residual_en=self.args.Residual_Set_list #[4,3] 4个子网络，每个3层残差


            self.model = Multi_scale2(
                sub_layer_number = np.array(sub_layer_number),
                layer_set = np.array(layer_set[0]),#实际是4个list，每个list是一个子网络的神经元
                act_set = np.array(multi_net_act),
                ini_set = np.array(multi_init_weight),
                residual= residual_en[0],
                scale_number=scale_omegas_coeff,
                scale_learn=self.args.Learn_scale#scale 学习
            )
        if self.args.model == "fnn":
            layer_set = self.args.Layer_Set_list#[1, 10, 10, 10, 1]
            residual_en = self.args.Residual_Set_list
            activation_set=self.args.Act_Set_list

            self.model = Single_MLP(
                input_size=layer_set[0],
                layer_set= layer_set,
                use_residual= residual_en,
                activation_set= np.array(activation_set)
            )

        return self.model

    def _Check(self):
        # 检查读取路径
        if self.args.PDE == False:
            if not os.path.exists(self.args.Train_Dataset):
                raise FileNotFoundError("Train_Dataset not found")

        # 检查保存路径, 如果没有就创建一个
        if not os.path.exists(self.args.Save_Path):
            os.makedirs(self.args.Save_Path)

    def _update_loss_record(self, epoch,train_loss,type=None,**kwargs):

        # 画loss的值
        if epoch % self.args.fig_record_interve == 0:
            if type!="deepxde":
                valid_loss=self._Valid(epoch=epoch,num_epochs=self.args.epoch)
                test_loss =self._Test4Save(epoch=epoch)
                record = np.array([[epoch, train_loss, valid_loss, test_loss]])
            #deepxde 在外面算test loss包括pde loss/data loss/ bc loss
            elif type=="deepxde":
                pde_loss=kwargs["pde_loss"]
                bc_loss=kwargs["bc_loss"]
                data_loss=kwargs["data_loss"]
                train_loss=train_loss.detach().cpu().numpy()
                test_loss=kwargs["test_loss"]
                test_data=kwargs["test_data"]
                sub_omega=kwargs["sub_omega"]
            
                record = np.array([[epoch, train_loss,test_loss,pde_loss,bc_loss,data_loss]])
            
            
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
            self._save4plot(epoch, test_loss,test_data=test_data,type="deepxde",sub_omega=sub_omega)

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

        # 获取模型预测
        pred = self.model(torch.from_numpy(x_test).float().to(self.device)).detach().cpu().numpy()
    

        if x_test.shape[-1] == 1: #[500,1]
            # analyzer
            analyzer = Analyzer4scale(model=self.model, d=1,
                                    scale_coeffs=self.args.Scale_Coeff)
            fig,axes=self.plot.plot_1d(nrow=3,ncol=3,
                                loss_record=loss_record_npy,
                                analyzer=analyzer,
                                x_test=x_test,
                                y_true=y_true,
                                pred=pred,
                                epoch=epoch,
                                avg_test_loss=avg_test_loss,
                                contribution_record=self.args.Con_record,)

        elif x_test.shape[-1] == 2: #[5000,2]
            # analyzer
            analyzer = Analyzer4scale(model=self.model,d=2,scale_coeffs=self.args.Scale_Coeff)
            fig, axes,cb_list = self.plot.plot_2d(nrow=4,ncol=3,
                                        loss_record=loss_record_npy,
                                        analyzer=analyzer,
                                        pred=pred,
                                        epoch=epoch,
                                        avg_test_loss=avg_test_loss,
                                        contribution_record=self.args.Con_record,
                                        solver=self.solver,
                                        model=self.model,
                                        contribution_record_path=self.args.Con_Record_Path,
                                        omega_value_path=self.args.Omega_Record_Path,
                                        omega=kwargs["sub_omega"])

        
        # 保存整个图表
        fig.savefig('{}/combined_loss_{}.png'.format(self.args.Save_Path, epoch),
                    bbox_inches='tight', format='png')
        
        for cb in cb_list:
            cb.remove()

        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[3].clear()
        axes[7].cla()
        axes[8].cla()

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

   
    def Train_XDE(self):



        #Copy the sover and agent to save
        des_file=self.args.Save_Path
        soure_file_solver=self.args.PDE_py
        soure_file_agent=self.args.Agent_py
        copy(soure_file_solver,des_file)
        copy(soure_file_agent,des_file)
        print("save solevr and agent")


        self.model = self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(),self.args.lr)
        
        print("train_data_penalty:{}".format(self.args.penalty_data))
        print("train_pde_penalty:{}".format(self.args.penalty_pde))
        print("train_bc_penalty:{}".format(self.args.penalty_boun))


                    
        for epoch in range(self.epoch):
            
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
            #pde loss
            train_pde_loss=self.solver.pde_loss(net=self.model,pde_data=pde_data)
            #bc loss
            train_bc_loss=self.solver.bc_loss(net=self.model,data=train_data)
            #data loss
            train_data_loss=self.solver.data_loss(net=self.model,data=pde_data)
            train_loss =  self.args.penalty_pde*train_pde_loss
            train_loss += self.args.penalty_boun*train_bc_loss


            print("train_data_loss",train_data_loss)
                                
            #train
            optimizer.zero_grad()
            train_loss.backward()
      
            #clip -2范数-长度
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()


            #plot
            if epoch % 1== 0:
                test=self.solver.data.test_x
                torch_test=torch.from_numpy(test).float().to(self.device)

                data_loss = self.solver.data_loss(net=self.model, data=pde_data)
                pde_loss=self.solver.pde_loss(net=self.model,pde_data=pde_data)
                bc_loss=self.solver.bc_loss(net=self.model,data=train_data)
                #test loss only see the data loss
                test_loss=bc_loss+pde_loss+data_loss 
                print(f"epoch={epoch},testloss{test_loss:.6f} pde_loss:{pde_loss:.6f},bc{bc_loss:.6f},data_loss:{data_loss:.6f}",flush=True)
                
            self._update_loss_record(   epoch, train_loss=train_loss,type="deepxde",
                                        pde_loss=pde_loss.item(),
                                        bc_loss=bc_loss.item(),
                                        data_loss=data_loss.item(),
                                        test_loss=test_loss.item(),
                                        test_data=torch_test,
                                        sub_omega=self.model.sub_omegas.detach().cpu().numpy())
                
    def Do_Expr(self):

        if self.args.PDE == "self_PDE":
            self.Train_PDE()
        elif self.args.PDE == "deepxde":
            self.Train_XDE()
        else:#fitting
            self.Train()
        print("we have done the expr")

if __name__=="__main__":
    pass