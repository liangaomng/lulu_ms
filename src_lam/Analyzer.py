
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from captum.attr import LayerConductance
from abc import abstractmethod


class Analyzer():
    def __init__(self,model,**kwagrs):
        self.model=model
        self.model.eval()
    @abstractmethod
    def plot_contributions(self):
        pass
class Analyzer4scale(Analyzer):
    def __init__(self,model,d,**kwagrs):
        super().__init__(model,**kwagrs)
        self.contributions=None
        self.scale_coeffs=kwagrs["scale_coeffs"]
        if d==1:
            self.input_tensor=torch.tensor([[1]])
        elif d==2:
            self.input_tensor=torch.tensor([[1,1]])

    def _analyze_scales(self,
                        baseline=-1,
                        n_steps=1000,
                        target=0)->list:
        scales_contribution = []
        input_tensor=self.input_tensor
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_tensor=input_tensor.float().to(device)
        for i, scale in enumerate(self.model.Multi_scale):
            layer_conductance = LayerConductance(self.model, scale)
            #target表示输出为层为第一个的神经元
            cond = layer_conductance.attribute(input_tensor, baselines=baseline, n_steps=n_steps, target=target)
            scales_contribution.append(cond.item())
        return scales_contribution
    def plot_contributions(self,ax=None,fig=None,cmap=plt.cm.Greens):

        self.contributions=self._analyze_scales()
        # 归一化 scale_coeffs 以便用于颜色映射
        norm = Normalize(vmin=min(self.scale_coeffs),
                            vmax=max(self.scale_coeffs))
        normed_coeffs = norm(self.scale_coeffs)
        if ax is None:
            raise ValueError("ax must be specified")

        if cmap=="Green":
            cmap = plt.cm.Greens
        elif cmap=="Blue":
            cmap = plt.cm.Blues
        elif cmap=="Purple":
            cmap = plt.cm.Purples
        for i, (contrib, coeff_norm) in enumerate(zip(self.contributions, normed_coeffs)):
            ax.bar(i, contrib, color=cmap(coeff_norm),
                    label=f'Scale {i}: Coeff {self.scale_coeffs[i]}')
            ax.text(i, contrib, f'{self.scale_coeffs[i]}',
                    ha='center', va='bottom',fontsize=7)

        # 添加颜色条
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        #fig.colorbar(sm, ax=ax)  # 为子图ax添加颜色条
        # 设置图表标题和轴标签
        ax.set_title('Contribution per Scale')
        ax.set_xlabel('Scale Index')
        ax.set_ylabel('Contribution',fontsize=10)
        ax.grid(True)
        ax.legend(loc='best',fontsize=6)




