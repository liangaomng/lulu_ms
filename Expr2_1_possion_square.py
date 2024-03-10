import argparse
from src_lam.Agent import Expr_Agent
from src_PDE.Heat import PDE_HeatData
from src_PDE.Helmholtz import PDE_HelmholtzData
from src_PDE.Poisson import PDE_PossionData

import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pytorch")
    parser.add_argument('--expr_set_path', type=str, help='expr_set_path')#"Expr2d/Expr_9.xlsx"
    parser.add_argument('--pde_name', type=str, help='pde_name')
    parser.add_argument('--shape', type=str, default="sqaure",help='shape')
    parser.add_argument('--freq', default=1.0,type=float, help='heat_freq')
    parser.add_argument('--mu', default=1.0,type=float, help='poisson_mu')
    parser.add_argument('--epoch', default=10000,type=int, help='epochs')
    parser.add_argument('--fig_record_intereve', default=500,type=int, help='fig save')
    
    
    args = parser.parse_args()
    print("args",args)

    #expr
    np.random.seed(1234)
   
    if args.pde_name == "Poisson":
        print("Poisson")
        solver= PDE_PossionData(mu=args.mu,shape=args.shape)
       
    
    print("PDE_Data:",solver)
    expr = Expr_Agent(
        pde_task="deepxde",
        solver=solver,
        Read_set_path=args.expr_set_path,
        compile_mode=False,
        train_epoch=args.epoch,
        fig_save_interve=args.fig_record_intereve,)

    expr.Do_Expr()


